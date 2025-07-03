
#include "bandwidth_calc_ptx_string.h"

#include "MemoryBandwidth.h"
#include "lwml.h"
#include "lwca.h"
#include "logging.h"
#include "PluginStrings.h"
#include "MemoryBandwidthShared.h"
#include "float.h"
#include "PluginCommon.h"


/*****************************************************************************/
#ifndef MIN
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#endif

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

/*****************************************************************************/

#define NUM_ELEMENTS (67108864)

const int g_blkszs[4] = { BLK0, BLK1, BLK2, BLK3 };
const int g_strids[8] = { STR0, STR1, STR2, STR3, STR4, STR5, STR6, STR7 };

/*****************************************************************************/

int perform_hbmem_bandwidth_test_on_gpu(int init_value, int elements, int times, membw_device_p gpu);

/*****************************************************************************/
MemoryBandwidth::MemoryBandwidth(TestParameters *testParameters, Plugin *plugin) : m_lwmlInitialized(0),
                                                                                   m_shouldStop(0),
                                                                                   m_dcgmRecorder(0),
																				   m_Ndevices(0), m_init_value(-1),
																				   m_times(100),
																				   m_elements(NUM_ELEMENTS)
{
    m_testParameters = testParameters;

    if (!m_testParameters)
    {
        throw std::runtime_error("Null testParameters passed in");
    }

    m_plugin = plugin;

    /* Initialize all member variables. They will be set later */
    m_minBandwidth = testParameters->GetDouble(MEMBW_STR_MINIMUM_BANDWIDTH);
    m_sbeFailureThreshold = testParameters->GetDouble(MEMBW_STR_SBE_ERROR_THRESHOLD);
    memset(&m_device[0], 0, sizeof(m_device));
}

/*****************************************************************************/
MemoryBandwidth::~MemoryBandwidth()
{
    /* Just call our cleanup function */
    Cleanup();
}

/*****************************************************************************/
void MemoryBandwidth::Cleanup(void)
{
    /* This code should be callable multiple times since exit paths and the
     * destructor will call this */

    LWresult lwSt;

    int deviceIdx;
    membw_device_p gpu = 0;

    for (deviceIdx = 0; deviceIdx < m_Ndevices; deviceIdx++)
    {
        gpu = &m_device[deviceIdx];

        /* Restore device settings */
        if (gpu->lwvsDevice)
        {
            gpu->lwvsDevice->RestoreState();
            delete(gpu->lwvsDevice);
        }
    }

    /* Do not delete m_testParameters. We don't own it */
    if (m_dcgmRecorder)
    {
        delete(m_dcgmRecorder);
        m_dcgmRecorder = 0;
    }

    /* Unload our lwca context for each gpu in the current process. We enumerate all GPUs because
       lwca opens a context on all GPUs, even if we don't use them */
    for (deviceIdx = 0; deviceIdx < m_Ndevices; deviceIdx++)
    {
        gpu = &m_device[deviceIdx];

        if (gpu->lwModule)
        {
            lwModuleUnload(gpu->lwModule);
            gpu->lwModule = 0;
        }

        /* Unload our context and reset the default context so that the next plugin gets a clean state */
        lwSt = lwCtxDestroy(gpu->lwContext);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwCtxDestroy", lwSt, gpu->dcgmDeviceIndex);
        }
        gpu->lwContext = 0;

        lwSt = lwDevicePrimaryCtxReset(gpu->lwDevice);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDevicePrimaryCtxReset", lwSt, gpu->dcgmDeviceIndex);
        }
    }

    PRINT_DEBUG("", "Cleaned up lwca contexts");

    m_Ndevices = 0;

    if (m_lwmlInitialized)
    {
        lwmlShutdown();
    }
    m_lwmlInitialized = 0;
}

/*****************************************************************************/
int MemoryBandwidth::LoadLwdaModule(membw_device_p gpu)
{
    LWresult lwSt;
    int i;

    lwSt = lwCtxSetLwrrent(gpu->lwContext);
    if(lwSt)
    {
        PRINT_ERROR("%d", "lwCtxSetLwrrent failed. lwSt: %d", lwSt);
        return -1;
    }

    /* Load our lwca module so we can find all of the functions */
    lwSt = lwModuleLoadData(&gpu->lwModule, (const char *)bandwidth_calc_ptx_string);
    if(lwSt)
    {
        PRINT_ERROR("%d", "Unable to load lwca module bandwidth_calc_ptx_string. lwSt: %d", lwSt);
        return -1;
    }

    /* Load functions from our lwca module */
    lwSt = lwModuleGetFunction(&gpu->lwFuncSetArray, gpu->lwModule, set_array_func_name);
    if(lwSt)
    {
        PRINT_ERROR("%s %d", "Unable to load lwca function %s. lwSt: %d", set_array_func_name, lwSt);
        return -1;
    }

    lwSt = lwModuleGetFunction(&gpu->lwFuncStreamTriadCleanup, gpu->lwModule, stream_triad_cleanup_func_name);
    if(lwSt)
    {
        PRINT_ERROR("%s %d", "Unable to load lwca function %s. lwSt: %d", stream_triad_cleanup_func_name, lwSt);
        return -1;
    }

    for(i = 0; i < NUM_STREAM_TRIADS; i++)
    {
        int st, loadPer, blkSize, blkStride, loadtype;
        st = sscanf(streamTriadFuncs[i], "_Z12STREAM_TriadILi%dELi%dELi%dELi%dEEvPKiS1_Piii", &loadPer, &blkSize, &blkStride, &loadtype);
        if(st != 4)
        {
            PRINT_ERROR("%d %s", "Only matched %d items in %s", st, streamTriadFuncs[i]);
            continue;
        }

        loadPer--; //1-4 -> 0-3
        if(loadPer < 0 || loadPer > 3)
            continue;

        if(blkSize >= BLK3)
            blkSize = 3;
        else if(blkSize >= BLK2)
            blkSize = 2;
        else if(blkSize >= BLK1)
            blkSize = 1;
        else
            blkSize = 0;

        if(blkStride >= STR7)
            blkStride = 7;
        else if(blkStride >= STR6)
            blkStride = 6;
        else if (blkStride >= STR5)
            blkStride = 5;
        else if (blkStride >= STR4)
            blkStride = 4;
        else if (blkStride >= STR3)
            blkStride = 3;
        else if (blkStride >= STR2)
            blkStride = 2;
        else if (blkStride >= STR1)
            blkStride = 1;
        else
            blkStride = 0;

        if(loadtype < 0 || loadtype > 3)
            continue;

        lwSt = lwModuleGetFunction(&gpu->lwFuncStreamTriad[loadtype][blkStride][loadPer][blkSize], gpu->lwModule,
                                   streamTriadFuncs[i]);
        if(lwSt)
        {
            PRINT_ERROR("%s %d %d", "Unable to load lwca function %s (i = %d). lwSt: %d", streamTriadFuncs[i], i, lwSt);
            return -1;
        }

        PRINT_DEBUG("%s %d %d %d %d", "Loaded function %s as lt %d, bstride %d, lp %d, blkSize %d",
                    streamTriadFuncs[i], loadtype, blkStride, loadPer, blkSize);
    }

    return 0;
}

/*****************************************************************************/
int MemoryBandwidth::LwdaInit(void)
{
    int i, j, st, deviceIdx, count, valueSize;
    LWresult lwSt;
    membw_device_p gpu = 0;
    unsigned int hostAllocFlags = 0;

    lwSt = lwDeviceGetCount(&count);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceGetCount", lwSt, 0, 0, false);
        return -1;
    }

    /* Do per-device initialization */
    for (deviceIdx = 0; deviceIdx < m_Ndevices; deviceIdx++)
    {
        gpu = &m_device[deviceIdx];

        lwSt = lwCtxCreate(&gpu->lwContext, 0, gpu->lwDevice);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwCtxCreate", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }

        lwSt = lwDeviceGetAttribute(&gpu->maxThreadsPerMultiProcessor, LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, gpu->lwDevice);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceGetAttribute", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
        PRINT_DEBUG("%d", "LW_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: %d", gpu->maxThreadsPerMultiProcessor);

        lwSt = lwDeviceGetAttribute(&gpu->multiProcessorCount, LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, gpu->lwDevice);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceGetAttribute", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
        PRINT_DEBUG("%d", "LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: %d", gpu->multiProcessorCount);

        lwSt = lwDeviceGetAttribute(&gpu->sharedMemPerMultiprocessor, LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, gpu->lwDevice);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceGetAttribute", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
        PRINT_DEBUG("%d", "LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: %d", gpu->sharedMemPerMultiprocessor);

        /* The modules must be loaded after we've created our contexts */
        st = LoadLwdaModule(gpu);
        if (st)
        {
            PRINT_ERROR("%d", "LoadLwdaModule failed with %d", st);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
int MemoryBandwidth::LwmlInit(const std::vector<unsigned int> &gpuList)
{
    int i, st, gpuListIndex;
    lwmlReturn_t lwmlSt;
    lwmlEnableState_t autoBoostEnabled, defaultAutoBoostEnabled;
    membw_device_p membwDevice = 0;
    char buf[256] = {0};
    lwmlPciInfo_t pciInfo;
    LWresult lwSt;
    lwvsReturn_t lwvsReturn;

    lwmlSt = lwmlInit();
    if (lwmlSt != LWML_SUCCESS)
    {
        lwvsCommon.errorMask |= SMPERF_ERR_GENERIC;
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlInit", lwmlErrorString(lwmlSt));
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        m_plugin->AddError(d);
        return -1;
    }

    /* Need to call lwInit before we call lwDeviceGetByPCIBusId */
    lwSt = lwInit(0);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwInit", lwSt, 0, 0, false);
        return -1;
    }

    for (gpuListIndex = 0; gpuListIndex < gpuList.size(); gpuListIndex++)
    {
        membwDevice = &m_device[m_Ndevices];

        membwDevice->dcgmDeviceIndex = gpuList[gpuListIndex];

        membwDevice->lwvsDevice = new LwvsDevice(m_plugin);
        st = membwDevice->lwvsDevice->Init(membwDevice->dcgmDeviceIndex);
        if (st)
        {
            return -1;
        }

        membwDevice->lwmlDevice = membwDevice->lwvsDevice->GetLwmlDevice();

        /* Resolve lwca device index from lwml device index */
        lwmlSt = lwmlDeviceGetPciInfo(membwDevice->lwmlDevice, &pciInfo);
        if (lwmlSt != LWML_SUCCESS)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWML_API, d, "lwmlDeviceGetPciInfo", lwmlErrorString(lwmlSt));
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            m_plugin->AddErrorForGpu(membwDevice->dcgmDeviceIndex, d);
            return -1;
        }

        lwSt = lwDeviceGetByPCIBusId(&membwDevice->lwDevice, pciInfo.busId);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceGetByPCIBusId", lwSt, gpuList[gpuListIndex], 0, true);
            return -1;
        }

        /* At this point, we consider this GPU part of our set */
        m_Ndevices++;
    }

    return 0;
}

/*****************************************************************************/
int MemoryBandwidth::Run(const std::vector<unsigned int> &gpuList)
{
    int st;
    bool testPassed;
    bool outputStats = true;
    unsigned int timeCount = 0;
    timelib64_t stopTime;
    timelib64_t startTime = timelib_usecSince1970();

    st = LwmlInit(gpuList);
    if (st)
    {
        Cleanup();
        return -1;
    }

    st = LwdaInit();
    if (st)
    {
        Cleanup();
        return -1;
    }

    /* Create the stats collection */
    if (!m_dcgmRecorder)
    {
        m_dcgmRecorder = new DcgmRecorder();
    }
    m_dcgmRecorder->Init(lwvsCommon.dcgmHostname);

    /* Is binary logging enabled for this stat collection? */
    std::string logFileName = m_testParameters->GetString(PS_LOGFILE);
    int logFileType = (int)m_testParameters->GetDouble(PS_LOGFILE_TYPE);
    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_CLOCK_THROTTLE_REASONS);
    // Allow 5 minutes for test duration
    m_dcgmRecorder->AddWatches(fieldIds, gpuList, false, "memory_bandwidth_field_group", "memory_bandwidth_group",
                               300.0);

    /* Wait for all workers to finish */
    try
    {
        /* Run test here */
		for (int i = 0; i < m_Ndevices; i++)
		{
            st = PerformBandwidthTestOnGpu(m_init_value, m_elements, m_times, &m_device[i]);
            if (st)
            {
                PRINT_ERROR("%d", "Stopping after PerformBandwidthTestOnGpu returned %d", st);
                break;
            }

            m_plugin->progressOut->updatePluginProgress((unsigned int)(i * 100)/m_Ndevices, false);
		}

    }
    catch (const std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught runtime_error %s", e.what());
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        m_plugin->AddError(d);
        m_plugin->SetResult(LWVS_RESULT_FAIL);
        Cleanup();
        throw;
    }
    stopTime = timelib_usecSince1970();

    m_plugin->progressOut->updatePluginProgress(timeCount, true);
    /* Don't check pass/fail if early stop was requested */
    if (main_should_stop)
    {
        Cleanup();
        return 1; /* Caller will check for main_should_stop and set the test skipped */
    }

    /* Set pass/failed status */
    testPassed = CheckPassFail(startTime, stopTime);

    if (testPassed && lwvsCommon.statsOnlyOnFail)
    {
        outputStats = false;
    }

    /* Should we write out a log file of stats? */
    if (logFileName.size() > 0 && outputStats)
    {
        st = m_dcgmRecorder->WriteToFile(logFileName, logFileType, startTime);
        if (st)
        {
            PRINT_ERROR("%s", "Unable to write to log file %s", logFileName.c_str());
            std::string error = "There was an error writing test statistics to file '";
            error += logFileName + "'.";
            m_plugin->AddInfo(error);
        }
    }

    Cleanup();
    return 0;
}

/*****************************************************************************/
int MemoryBandwidth::CheckLwmlEvents(membw_device_p mbDevice, std::vector<DcgmError> &errorList,
                             timelib64_t startTime, timelib64_t stopTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = 0;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    if (DCGM_FP64_IS_BLANK(m_sbeFailureThreshold) == 0)
    {
        dcgmTimeseriesInfo_t dti;
        // Only evaluate SBEs if a threshold was requested
        fieldIds.push_back(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
        memset(&dti, 0, sizeof(dti));
        dti.isInt = true;
        dti.val.i64 = static_cast<long long>(m_sbeFailureThreshold);
        failureThresholds.push_back(dti);

        // Set a threshold of 0 for the other two values
        dti.val.i64 = 0;
        for (int i = 0; i < 2; i++)
            failureThresholds.push_back(dti);
        thresholdsPtr = &failureThresholds;
    }

    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);

    return m_dcgmRecorder->CheckErrorFields(fieldIds, thresholdsPtr, mbDevice->dcgmDeviceIndex, 
                                            errorList, startTime, stopTime);
}

/*****************************************************************************/
bool MemoryBandwidth::CheckPassFailSingleGpu(membw_device_p device, std::vector<DcgmError> &errorList, 
                                             timelib64_t startTime, timelib64_t endTime)
{
    int st;
    char buf[256];
    bool passed = true;

    // Verify bandwidth values
    for (int j = 0; j < SI_TEST_COUNT; j++)
    {
        snprintf(buf, sizeof(buf), "Device %d's bandwidth was %.2f GB/s",
                    device->dcgmDeviceIndex, device->mbPerSec[j] / 1000.0);
        m_plugin->AddInfo(std::string(buf)); // TODO: Do we want to report this to the user as verbose info?

        if (device->mbPerSec[j] < m_minBandwidth)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_BANDWIDTH, d, device->dcgmDeviceIndex,
                                      device->mbPerSec[j] / 1000.0, m_minBandwidth / 1000.0, j);
            errorList.push_back(d);
            passed = false;
        }
    }
    
    st = m_dcgmRecorder->CheckForThrottling(device->dcgmDeviceIndex, startTime, errorList);
    if (st != DR_SUCCESS)
    {
        passed = false;
    }

    st = CheckLwmlEvents(device, errorList, startTime, endTime);
    if (st != DR_SUCCESS)
    {
        passed = false;
    }
    return passed;
}
/*****************************************************************************/
bool MemoryBandwidth::CheckPassFail(timelib64_t startTime, timelib64_t stopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorListAllGpus.
     */
    m_dcgmRecorder->GetLatestValuesForWatchedFields(0, errorListAllGpus);

    for (int i = 0; i < m_Ndevices; i++)
    {
        errorList.clear();
        passed = CheckPassFailSingleGpu(m_device + i, errorList, startTime, stopTime);
        if (passed)
        {
            m_plugin->SetResultForGpu(m_device[i].dcgmDeviceIndex, LWVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            m_plugin->SetResultForGpu(m_device[i].dcgmDeviceIndex, LWVS_RESULT_FAIL);
            // Log warnings for this gpu
            for (size_t j = 0; j < errorList.size(); j++)
            {
                m_plugin->AddErrorForGpu(m_device[i].dcgmDeviceIndex, errorList[j]);
            }
        }
    }

    for (size_t i=0; i < errorListAllGpus.size(); i++)
    {
        m_plugin->AddError(errorListAllGpus[i]);
    }

    return allPassed;
}

/*****************************************************************************/
//test kernels routine
int MemoryBandwidth::TestKernels(membw_device_p gpu, int kern, int i, int j, int s, int inshmem, int ldtp , REAL init_value)
{
    #define NUM (67108864)

    int num_elements=NUM;
    int ntimes = 100;
    int steps=0;

    #define SI_MAX_GPUS 16

    int shmem = inshmem;
    if(inshmem<1)
        shmem=1;
    if(inshmem>(48*1024))
        shmem=48*1024;
    if( ((kern==0)||(kern==2)||(kern==4)||(kern==5)) && ldtp!=0)
        return 0;
    if( kern==1 && ldtp>=2 )
        return 0;
    int lp = i+1;
    int blksz = g_blkszs[j];
    int strd = g_strids[s];
    int grdsz = ((num_elements)/(lp*blksz*strd))*strd;
    int gy = (grdsz+65534)/65535;
    int gx = (gy == 0) ? 1 : grdsz/gy;
    int iter=0;
    int extra=grdsz-gx*gy;
    int temp_gy = gy;
    int temp_gx = gx;


    while(extra>0&&iter<1000){
        temp_gy++;
        temp_gx = grdsz/temp_gy;
        if(extra>grdsz-temp_gx*temp_gy){
            gy = temp_gy;
            gx = temp_gx;
            extra = grdsz-temp_gx*temp_gy;
        }
        iter++;
    }
    dim3 testBlock(blksz);
    dim3 testGrid(gx,gy,1);
    int extra_elements = num_elements%(lp*blksz*strd)+extra*(lp*blksz*strd);

    LWresult lwSt;

    lwSt = lwEventRecord( gpu->lwda_timer[0][0], 0 );
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwEventRecord", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    /* Declaring these out here because of compiler warnings */
    void *kernelParams[6];
    int two = 2;

    switch( kern ){
        case SI_TEST_TRIAD:
            if(!gpu->lwFuncStreamTriad[ldtp][s][i][j])
            {
                PRINT_ERROR("%d %d %d %d", "Missing lwca function %d %d %d %d",
                            ldtp, s, i, j);
                return -1;
            }

            kernelParams[0] = &gpu->d_a;
            kernelParams[1] = &gpu->d_b;
            kernelParams[2] = &gpu->d_c[SI_TEST_TRIAD];
            kernelParams[3] = &two;
            kernelParams[4] = &num_elements;

            //Runtime implementation:
            //Triad_f[ldtp][s][i][j]<<<testGrid,testBlock,shmem,gpu->streamA>>>
            //  (gpu->d_a, gpu->d_b, gpu->d_c[SI_TEST_TRIAD], 2,  num_elements);
            lwSt = lwLaunchKernel(gpu->lwFuncStreamTriad[ldtp][s][i][j],
                                  testGrid.x, testGrid.y, testGrid.z,
                                  testBlock.x, testBlock.y, testBlock.z,
                                  shmem, gpu->streamA, kernelParams, 0);
            if (lwSt)
            {
                LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
                return -1;
            }

            if(extra_elements>0)
            {
                dim3 testGrid2((extra_elements+129)/128);
                dim3 testBlock2(128);
                int diff = num_elements-extra_elements;
                kernelParams[0] = &gpu->d_a;
                kernelParams[1] = &gpu->d_b;
                kernelParams[2] = &gpu->d_c[SI_TEST_TRIAD];
                kernelParams[3] = &two;
                kernelParams[4] = &num_elements;
                kernelParams[5] = &diff;
                //Runtime implementations:
                //STREAM_Triad_cleanup<<<(extra_elements+129)/128,128,0,gpu->streamB>>>
                //(gpu->d_a, gpu->d_b, gpu->d_c[SI_TEST_TRIAD], 2,  num_elements,
                // num_elements-extra_elements);
                lwSt = lwLaunchKernel(gpu->lwFuncStreamTriadCleanup,
                                      testGrid2.x, testGrid2.y, testGrid2.z,
                                      testBlock2.x, testBlock2.y, testBlock2.z,
                                      0, gpu->streamB, kernelParams, 0);
                if (lwSt)
                {
                    LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
                    return -1;
                }
            }
            break;
        default:
            return 0;
    }
    lwSt = lwEventRecord( gpu->lwda_timer[1][0], 0 );
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwEventRecord", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    lwSt = lwCtxSynchronize();
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwCtxSynchronize", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    lwSt = lwEventElapsedTime( &gpu->ms_kern[kern], gpu->lwda_timer[0][0], gpu->lwda_timer[1][0] );
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwdaEventElapsedTime", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    gpu->perf_kern[kern] = 1.0e-3*1.0*(double)(num_elements)*gpu->byte_kern[kern]*(double)(sizeof(REAL))/gpu->ms_kern[kern];

    if(gpu->perf_kern[kern]>gpu->opt_perf_kern[kern])
    {
        gpu->i_kern[kern] = i;
        gpu->j_kern[kern] = j;
        gpu->s_kern[kern] = s;
        gpu->opt_perf_kern[kern] = gpu->perf_kern[kern];
        gpu->best_lp_kern[kern]=lp;
        gpu->best_strd_kern[kern]=strd;
        gpu->best_lt_kern[kern] = ldtp;
        gpu->block_kern[kern].x = blksz;
        gpu->block_kern[kern].y = 1;
        gpu->block_kern[kern].z = 1;
        gpu->grid_kern[kern].x = gx;
        gpu->grid_kern[kern].y = gy;
        gpu->grid_kern[kern].z = 1;
        gpu->shmem_kern[kern] = shmem;
        gpu->extra_kern[kern] = extra_elements;
        switch( kern ){
            case SI_TEST_TRIAD:
                PRINT_DEBUG("%d %d %d %f %d %d %d %d %d %d", "New best i %d, j %d, s %d, opk %f, ldtp %d, blkszz %d, gridX %d, gridY %d, shmem %d, extra %d",
                            i, j, s, gpu->perf_kern[kern], ldtp, blksz, gx, gy, shmem, extra_elements);
                gpu->lwFuncTriadOptimal = gpu->lwFuncStreamTriad[ldtp][s][i][j];
                break;
        }
    }
    steps++;
    return 0;
}

/*****************************************************************************/
int MemoryBandwidth::InitializeDeviceMemory(membw_device_p gpu, int init_value, int element_count)
{
    int i;
    LWresult lwSt;

    /* Compute exelwtion configuration */
    dim3 dimBlock(512);
    //dim3 dimGrid((num_elements+ (dimBlock.x - 1)) / dimBlock.x); //This was previously unused
    dim3 dimGrid(gpu->multiProcessorCount);
    int sharedMemBytes = 0;
    LWstream stream = LW_STREAM_PER_THREAD;
    void *kernelParams[3] = {&gpu->d_a, &init_value, &element_count};

    /* Initialize memory on the device */
    //Runtime function:
    //set_array<<<gpu->multiProcessorCount,dimBlock>>>(gpu->d_a, init_value, num_elements);
    lwSt = lwLaunchKernel(gpu->lwFuncSetArray, dimGrid.x, dimGrid.y, dimGrid.z,
                          dimBlock.x, dimBlock.y, dimBlock.z, sharedMemBytes,
                          stream, kernelParams, 0);

    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    kernelParams[0] = &gpu->d_b;
    //set_array<<<gpu->multiProcessorCount,dimBlock>>>(gpu->d_b, init_value, num_elements);
    lwSt = lwLaunchKernel(gpu->lwFuncSetArray, dimGrid.x, dimGrid.y, dimGrid.z,
                          dimBlock.x, dimBlock.y, dimBlock.z, sharedMemBytes,
                          stream, kernelParams, 0);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }


    for (i = 0; i < SI_TEST_COUNT; i++)
    {
        kernelParams[0] = &gpu->d_c[i];
        //set_array<<<gpu->multiProcessorCount,dimBlock>>>(gpu->d_c[i], init_value, num_elements);
        lwSt = lwLaunchKernel(gpu->lwFuncSetArray, dimGrid.x, dimGrid.y, dimGrid.z,
                          dimBlock.x, dimBlock.y, dimBlock.z, sharedMemBytes,
                          stream, kernelParams, 0);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
/*
 * Callwlates the bandwidth for the specified GPU
 */
int MemoryBandwidth::PerformBandwidthTestOnGpu(int init_value, int element_count, int times, membw_device_p gpu)
{
    static REAL   bytes[SI_TEST_COUNT] = {
        3 * sizeof(REAL) * NUM,
        };

    LWresult lwSt;
    int i;

    lwSt = lwCtxSetLwrrent(gpu->lwContext);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwCtxSetLwrrent", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    PRINT_INFO("%d", "Running on Device %d", gpu->lwDevice);

	int num_elements = element_count;
	int ntimes = times;

    //printf(" STREAM Benchmark implementation in LWCA\n");
    //printf(" Array size (%s)=%d*%d (%d KB)\n", MK_STR(REAL), (int)num_elements, (int)sizeof(REAL), 8*(int)(num_elements*sizeof(REAL)/1024));

    for (i = 0; i < SI_TEST_COUNT; i++)
    {
        gpu->mintime[i] = FLT_MAX;
    }

    gpu->opt_perf_kern[SI_TEST_TRIAD] = 0.0;
    gpu->i_kern[SI_TEST_TRIAD] = 1;
    gpu->j_kern[SI_TEST_TRIAD] = 2;
    gpu->s_kern[SI_TEST_TRIAD] = 0;
    gpu->occ_kern[SI_TEST_TRIAD] = 1.0;
    gpu->best_lp_kern[SI_TEST_TRIAD]  = 2;
    gpu->best_strd_kern[SI_TEST_TRIAD] = 1;
    gpu->best_lt_kern[SI_TEST_TRIAD]  = 2;
    gpu->byte_kern[SI_TEST_TRIAD] = 3;
    gpu->shmem_kern[SI_TEST_TRIAD] = 0;

    /* Allocate memory on device */
    lwSt = lwMemAlloc((LWdeviceptr*)&gpu->d_a, sizeof(REAL)*num_elements);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemAlloc", lwSt, gpu->dcgmDeviceIndex, sizeof(REAL)*num_elements);
        return -1;
    }

    lwMemAlloc((LWdeviceptr*)&gpu->d_b, sizeof(REAL)*num_elements);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemAlloc", lwSt, gpu->dcgmDeviceIndex, sizeof(REAL)*num_elements);
        return -1;
    }

    for (i = 0; i < SI_TEST_COUNT; i++)
    {
        lwSt = lwMemAlloc((LWdeviceptr*)&gpu->d_c[i], sizeof(REAL)*num_elements);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwMemAlloc", lwSt, gpu->dcgmDeviceIndex, sizeof(REAL)*num_elements);
            return -1;
        }
    }

    lwSt = lwStreamCreate(&gpu->streamA, LW_STREAM_DEFAULT);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwStreamCreate", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    lwSt = lwStreamCreate(&gpu->streamB, LW_STREAM_DEFAULT);
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwStreamCreate", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }


    for (int k = 0; k < ntimes; k++)
    {
        for (int j = 0; j < SI_TEST_COUNT + 1; j++)
        {
            lwSt = lwEventCreate(&(gpu->lwda_timer[k][j]), LW_EVENT_DEFAULT);
            if (lwSt)
            {
                LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwEventCreate", lwSt, gpu->dcgmDeviceIndex);
                return -1;
            }
        }
    }

    /* Set all of the device memory to a particular value */
    int st = InitializeDeviceMemory(gpu, init_value, element_count);
    if (st)
    {
        PRINT_ERROR("%d", "InitializeDeviceMemory returned %d", st);
        return -1;
    }

    /* --- search for best kernels --- */

    //printf("\nOptimizing...\n");

    int SHM = gpu->sharedMemPerMultiprocessor/1024;

#ifndef BRUTE_FORCE
    for(int kern=0; kern < SI_TEST_COUNT; kern++)
    {

        for(int j=0; j<4; j++){
            for(int bxs=0; bxs<11; bxs++){

                int s = bxs - j; if(s<0 || s>7)
                    continue;
                int lvbl = (int)(gpu->occ_kern[kern]*(double)gpu->maxThreadsPerMultiProcessor)/(g_blkszs[j]);
                if(lvbl<1)
                    lvbl=1;
                int shmem = (SHM*1024)/(lvbl);
                TestKernels(gpu, kern, gpu->i_kern[kern], j, s, shmem, gpu->best_lt_kern[kern] , init_value);
            }
            for (int blocks=gpu->maxThreadsPerMultiProcessor/g_blkszs[j]; blocks>=1; blocks--){
                int bxs = gpu->j_kern[kern] + gpu->s_kern[kern];
                int s = bxs - j;
                if(s<0 || s>7)
                    continue;
                int shmem = (SHM*1024)/blocks;
                TestKernels(gpu, kern, gpu->i_kern[kern], j, s, shmem, gpu->best_lt_kern[kern] , init_value);
            }
            gpu->occ_kern[kern] = (SHM*1024.0/(double)gpu->shmem_kern[kern])*(double)g_blkszs[gpu->j_kern[kern]]/(double)gpu->maxThreadsPerMultiProcessor;
        }
        for(int i=0; i<4; i++){
            TestKernels(gpu, kern, i, gpu->j_kern[kern], gpu->s_kern[kern], gpu->shmem_kern[kern], gpu->best_lt_kern[kern] , init_value);
        }
        for (int ldtp=0; ldtp<4; ldtp++){
            TestKernels(gpu, kern, gpu->i_kern[kern], gpu->j_kern[kern], gpu->s_kern[kern], gpu->shmem_kern[kern], ldtp , init_value);
        }

        for (int di=-1; di<=1; di++){
            for (int j=0; j<3; j++){
                for (int ds=-1; ds<=1; ds++){
                    int i = gpu->i_kern[kern]+di;
                    if(i>3)
                        i = gpu->i_kern[kern] - 2;
                    if(i<0)
                        i = gpu->i_kern[kern] + 2;
                    int s = gpu->s_kern[kern]+ds;
                    if(s>7)
                        s = gpu->s_kern[kern] - 2;
                    if(s<0)
                        s = gpu->s_kern[kern] + 2;

                    for(int dshmem = -1; dshmem<=1; dshmem++){
                        int lvbl = (int)(gpu->occ_kern[kern]*(double)gpu->maxThreadsPerMultiProcessor)/(g_blkszs[j] + dshmem);
                        if(lvbl<1) lvbl=dshmem + 3;
                        int shmem = (SHM*1024)/(lvbl);
                        //if(shmem<=0) shmem = (SHM*1024)/(int(occ_kern[kern]*(double)maxThreadsPerMultiProcessor)/g_blkszs[j] + 2);
                        //printf("\n%d\n",shmem); getchar();

                        for(int ldtp=0; ldtp<3; ldtp++){
                            TestKernels(gpu, kern, i, j, s, shmem, ldtp , init_value);
                        }
                    }
                }
            }
        }
        gpu->occ_kern[kern] = (SHM*1024.0/(double)gpu->shmem_kern[kern])*(double)g_blkszs[gpu->j_kern[kern]]/(double)gpu->maxThreadsPerMultiProcessor;
        for(int j=0; j<4; j++){
            for(int bxs=0; bxs<11; bxs++){

                int s = bxs - j;
                if(s<0 || s>7)
                    continue;

                int lvbl = (int)(gpu->occ_kern[kern]*(double)gpu->maxThreadsPerMultiProcessor)/(g_blkszs[j]);
                if(lvbl<1) lvbl=1;
                int shmem = (SHM*1024)/(lvbl);
                TestKernels(gpu, kern, gpu->i_kern[kern], j, s, shmem, gpu->best_lt_kern[kern] , init_value);
            }
            for (int blocks=gpu->maxThreadsPerMultiProcessor/g_blkszs[j]; blocks>=1; blocks--){
                int bxs = gpu->j_kern[kern] + gpu->s_kern[kern];
                int s = bxs - j;  if(s<0 || s>7) continue;
                int shmem = (SHM*1024)/blocks;
                TestKernels(gpu, kern, gpu->i_kern[kern], j, s, shmem, gpu->best_lt_kern[kern] , init_value);
            }
            gpu->occ_kern[kern] = (SHM*1024.0/(double)gpu->shmem_kern[kern])*(double)g_blkszs[gpu->j_kern[kern]]/(double)gpu->maxThreadsPerMultiProcessor;
        }
        for(int i=0; i<4; i++){
            TestKernels(gpu, kern, i, gpu->j_kern[kern],  gpu->s_kern[kern],  gpu->shmem_kern[kern], gpu-> best_lt_kern[kern] , init_value);
        }
        for (int ldtp=0; ldtp<4; ldtp++){
            TestKernels(gpu, kern, gpu->i_kern[kern], gpu->j_kern[kern], gpu->s_kern[kern], gpu->shmem_kern[kern], ldtp , init_value);
        }
    }
#else
    for(int kern=0; kern < SI_TEST_COUNT; kern++){
        for (int i=0; i<4; i++){
            for (int j=0; j<4; j++){
                for (int s=0; s<8; s++){
                    int shmem;
                    for (int blocks=maxThreadsPerMultiProcessor/g_blkszs[j]; blocks*g_blkszs[j]>=gpu->maxThreadsPerMultiProcessor/5; blocks--){
                        for (int ldtp=0; ldtp<4; ldtp++){
                            shmem = (SHM*1024)/blocks;
                            TestKernels(gpu, kern, i, j, s, shmem, ldtp , init_value);
                        }
                    }
                }
            }
        }
        occ_kern[kern] = (SHM*1024.0/(double)gpu->shmem_kern[kern])*(double)g_blkszs[j_kern[kern]]/(double)gpu->maxThreadsPerMultiProcessor;
    }
#endif

    //printf("\n\nOptimization Complete... \n\n");

    /* Initialize memory on the device */
    st = InitializeDeviceMemory(gpu, init_value, element_count);
    if (st)
    {
        PRINT_ERROR("%d", "InitializeDeviceMemory returned %d", st);
        return -1;
    }

    #if 0
        double peakbw = 2.0*memclock*(gpu->lwdaDevProp.memoryBusWidth/8)/1000.0;
        printf("--- Best Triad ---  lt: %d lp: %d\t block: %d \t stride: %d \t shmem: %d \t occ: %4.1f%%\t GB/s: %4.1f\teff: %4.1f%%\n",
               best_lt_kern[SI_TEST_TRIAD],best_lp_kern[SI_TEST_TRIAD],block_kern[SI_TEST_TRIAD].x ,best_strd_kern[SI_TEST_TRIAD],shmem_kern[SI_TEST_TRIAD],
               100.0*occ_kern[SI_TEST_TRIAD],opt_perf_kern[SI_TEST_TRIAD]/1000.0,opt_perf_kern[SI_TEST_TRIAD]/1000.0/peakbw*100.0 );
    #endif

    lwSt = lwCtxSynchronize();
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceSynchronize", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }

    dim3 testGrid(gpu->grid_kern[SI_TEST_TRIAD]);
    dim3 testBlock(gpu->block_kern[SI_TEST_TRIAD]);
    int sharedMemBytes = gpu->shmem_kern[SI_TEST_TRIAD];
    int two = 2;
    void *kernelParams[6];

    kernelParams[0] = &gpu->d_a;
    kernelParams[1] = &gpu->d_b;
    kernelParams[2] = &gpu->d_c[SI_TEST_TRIAD];
    kernelParams[3] = &two;
    kernelParams[4] = &num_elements;

    //Triad_optimal<<<gpu->grid_kern[SI_TEST_TRIAD],gpu->block_kern[SI_TEST_TRIAD],gpu->shmem_kern[SI_TEST_TRIAD],gpu->streamA>>>
    //                (gpu->d_a, gpu->d_b, gpu->d_c[SI_TEST_TRIAD], 2,  num_elements);
    lwSt = lwLaunchKernel(gpu->lwFuncTriadOptimal, testGrid.x, testGrid.y, testGrid.z,
                          testBlock.x, testBlock.y, testBlock.z, sharedMemBytes,
                          gpu->streamA, kernelParams, 0);

    /* --- MAIN LOOP --- repeat test cases NTIMES times --- */
    //printf("press any key to run test...\n"); getchar();
    //printf("\nstarting test... \n\n");
    for (int k=0; k<ntimes; k++)
    {
        lwSt = lwEventRecord(gpu->lwda_timer[k][SI_TEST_TRIAD], 0);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwEventRecord", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
        if (!gpu->lwFuncTriadOptimal)
        {
            PRINT_ERROR("", "No optimal function found!");
            return -1;
        }

        kernelParams[0] = &gpu->d_a;
        kernelParams[1] = &gpu->d_b;
        kernelParams[2] = &gpu->d_c[SI_TEST_TRIAD];
        kernelParams[3] = &two;
        kernelParams[4] = &num_elements;

        //Triad_optimal<<<gpu->grid_kern[SI_TEST_TRIAD],gpu->block_kern[SI_TEST_TRIAD],gpu->shmem_kern[SI_TEST_TRIAD],gpu->streamA>>>
        //                (gpu->d_a, gpu->d_b, gpu->d_c[SI_TEST_TRIAD], 2,  num_elements);
        lwSt = lwLaunchKernel(gpu->lwFuncTriadOptimal, testGrid.x, testGrid.y, testGrid.z,
                                testBlock.x, testBlock.y, testBlock.z, sharedMemBytes,
                                gpu->streamA, kernelParams, 0);
        //lwSt = lwLaunchKernel(m_lwFuncStreamTriad[0][0][0][0], 0, 1, 1,
        //                        1, 1, 1, 0,
        //                        0, kernelParams, 0);
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }

        if (gpu->extra_kern[SI_TEST_TRIAD]>0)
        {
            dim3 testGrid2((gpu->extra_kern[SI_TEST_TRIAD]+129)/128);
            dim3 testBlock2(128);
            sharedMemBytes = 0;
            kernelParams[0] = &gpu->d_a;
            kernelParams[1] = &gpu->d_b;
            kernelParams[2] = &gpu->d_c[SI_TEST_TRIAD];
            kernelParams[3] = &two;
            kernelParams[4] = &num_elements;
            int diff = num_elements-gpu->extra_kern[SI_TEST_TRIAD];
            kernelParams[5] = &diff;

            //STREAM_Triad_cleanup<<<(gpu->extra_kern[SI_TEST_TRIAD]+129)/128,128,0,gpu->streamB>>>
            //(gpu->d_a, gpu->d_b, gpu->d_c[SI_TEST_TRIAD], 2,  num_elements,
            // num_elements-gpu->extra_kern[SI_TEST_TRIAD]);
            lwSt = lwLaunchKernel(gpu->lwFuncStreamTriadCleanup,
                                  testGrid2.x, testGrid2.y, testGrid2.z,
                                  testBlock2.x, testBlock2.y, testBlock2.z,
                                  sharedMemBytes, gpu->streamB, kernelParams, 0);
            if (lwSt)
            {
                LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwLaunchKernel", lwSt, gpu->dcgmDeviceIndex);
                return -1;
            }


        }
        lwSt = lwEventRecord( gpu->lwda_timer[k][SI_TEST_COUNT], 0 );
        if (lwSt)
        {
            LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwEventRecord", lwSt, gpu->dcgmDeviceIndex);
            return -1;
        }
    }

    lwSt = lwCtxSynchronize();
    if (lwSt)
    {
        LOG_LWDA_ERROR_FOR_PLUGIN(m_plugin, "lwDeviceSynchronize", lwSt, gpu->dcgmDeviceIndex);
        return -1;
    }
    //printf("test done ... \n\n");
    /*  --- SUMMARY --- */

    for (int k=1; k<ntimes; k++) /* note -- skip first iteration */
    {
        for (int j=0; j < SI_TEST_COUNT; j++)
        {
            float ms;
            lwEventElapsedTime( &ms, gpu->lwda_timer[k][j], gpu->lwda_timer[k][j+1] );
            gpu->avgtime[j] = gpu->avgtime[j] + ms/1000.0;
            gpu->mintime[j] = MIN(gpu->mintime[j], ms/1000.0);
            gpu->maxtime[j] = MAX(gpu->maxtime[j], ms/1000.0);
        }
    }

    for (int j=0; j < SI_TEST_COUNT; j++)
    {
        gpu->avgtime[j] = gpu->avgtime[j]/(double)(ntimes-1);
        double mbPerSec = 1.0E-06 * bytes[j]/gpu->mintime[j] * (double)num_elements/(double)NUM;
        gpu->mbPerSec[j] = mbPerSec;
    }


    /* Free memory on device */
    lwMemFree((LWdeviceptr)gpu->d_a);
    gpu->d_a = 0;
    lwMemFree((LWdeviceptr)gpu->d_b);
    gpu->d_b = 0;

    #if 0
    REAL *hmem = (REAL *)malloc(sizeof(REAL)*num_elements);

    static const char* label[SI_TEST_COUNT] = {"HBM Bandwidth: "};

    if (hmem && init_value != -1)
    {
        printf("\n");
        for (i = 0; i < SI_TEST_COUNT; i++) {
            printf("Checking result for %s: ", label[i]);
            lwMemcpyDtoH(hmem, (LWdeviceptr)gpu->d_c[i], sizeof(REAL)*num_elements);
            REAL gold = hmem[0];
            int j;
            for (j = 1; j < num_elements; j++) {
                //if (j < 256) printf("0x%08x%s", hmem[j], (j%8)?" ":"\n");
                if (hmem[j] == gold || hmem[j] == 0) {
                    // ok
                } else {
                    printf("failed to predict value, gold != mem[%d], 0x%x != 0x%x\n", j, gold, hmem[j]);
                    break;
                }
            }
            if (j == num_elements) {
                printf("all values equal 0x%x, passed!\n", gold);
            }
        }
    }
    #endif

    for (i = 0; i < SI_TEST_COUNT; i++)
    {
        lwMemFree((LWdeviceptr)gpu->d_c[i]);
        gpu->d_c[i] = 0;
    }

    return 0;
}

/*****************************************************************************/
