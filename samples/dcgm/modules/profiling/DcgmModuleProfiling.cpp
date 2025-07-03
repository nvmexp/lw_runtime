#include "DcgmModuleProfiling.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"
#include <sstream>
#include <algorithm>
#include "DcgmLopGpu.h"
#include "lwperf_target.h"
#include "lwperf_lwda_target.h"
#include "lwperf_host_priv_impl.h" //Don't include the target_priv_impl if you include the host one

/* Keep a backup of Perfworks's stub API as we're going to use it when we shut down */
LWPW_User_Api g_api_backup = g_api;

/*****************************************************************************/
DcgmModuleProfiling::DcgmModuleProfiling()
{
    memset(&m_stats, 0, sizeof(m_stats));
    m_watcherConnId = DCGM_CONNECTION_ID_NONE;
    m_taskRunnerFvBuffer = NULL;
#ifdef DEBUG
    m_bypassSkuCheck = true; /* Don't SKU-check internal-only debug builds */
#else //!DEBUG
    m_bypassSkuCheck = false;
#endif
    m_profilingIsPaused = false;

    m_cacheIntervalUsec = 0;
    
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModuleProfiling was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    mpGroupManager = LwcmHostEngineHandler::Instance()->GetGroupManager();
    if(!mpGroupManager)
    {
        const char *errorStr = "DcgmModuleProfiling was unable to find the group manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    /* Read any elwironmental variables before we set any */
    ReadElwironmentalVariables();

    /* Set any required elwironental variables before lwca and/or LOP is loaded */
    SetElwironmentalVariables();

    /* Initialize LOP */
    dcgmReturn_t dcgmReturn = InitLop();
    if(dcgmReturn != DCGM_ST_OK)
    {
        const char *errorStr = "DcgmModuleProfiling failed to initialize. See the logs.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }

    m_taskRunnerFvBuffer = new DcgmFvBuffer();

    /* Start our TaskRunner now that we've survived initialization */
    SetDebugLogging(false);
    int st = Start();
    if(st)
    {
        PRINT_ERROR("%d", "Got error %d from Start()", st);
        throw std::runtime_error("Unable to start a DcgmTaskRunner");
    }
}

/*****************************************************************************/
DcgmModuleProfiling::~DcgmModuleProfiling()
{
    /* Wait for our task runner to exit */
    StopAndWait(60000);

    LogStats(false, true);

    /* Clean up watches and other internal structures */
    for(unsigned int i = 0; i < m_gpus.size(); i++)
    {
        dcgm_module_prof_gpu_t *gpu = &m_gpus[i];

        /* Clear any previous watches for this GPU */
        ClearLopWatchesForGpu(gpu);

        if(gpu->lopGpu)
        {
            delete(gpu->lopGpu);
            gpu->lopGpu = NULL;
        }
    }

    if(m_taskRunnerFvBuffer)
    {
        delete(m_taskRunnerFvBuffer);
        m_taskRunnerFvBuffer = NULL;
    }

    mpCacheManager = 0; /* Not owned by us */
    mpGroupManager = 0; /* Not owned by us */

    /* Unload perfworks if it's loaded */
    if(g_api.hModPerfworks)
    {
        dlclose(g_api.hModPerfworks);
        g_api = g_api_backup;
    }
}

/*****************************************************************************/
void DcgmModuleProfiling::SetElwironmentalVariables(void)
{
   /* Set any elwironmental variables perfworks cares about */
   //lwosSetElw("", "1");
}

/*****************************************************************************/
void DcgmModuleProfiling::ReadElwironmentalVariables(void)
{
    int st;
    char buffer[64];

    memset(buffer, 0, sizeof(buffer));
    /* Read any elwironmental variables before we set any */
    
    st = lwosGetElw("__DCGM_PROF_NO_SKU_CHECK", buffer, sizeof(buffer));
    if(st >= 0)
    {
        PRINT_INFO("", "__DCGM_PROF_NO_SKU_CHECK was set. disabling SKU check");
        m_bypassSkuCheck = true;
    }
    else
        PRINT_DEBUG("", "__DCGM_PROF_NO_SKU_CHECK was NOT set.");
    
}

/*****************************************************************************/
void DcgmModuleProfiling::LogStats(bool logToStdout, bool logToDebugLog)
{
    std::stringstream ss;
    ss << "totalMetricsRead: " << m_stats.totalMetricsRead << ", ";
    ss << "numLopReadCalls: " << m_stats.numLopReadCalls << ", ";
    ss << "totalUsecInLopReads: " << m_stats.totalUsecInLopReads << ", ";

    double usecsPerLopRead = 0.0;
    if(m_stats.numLopReadCalls)
        usecsPerLopRead = m_stats.totalUsecInLopReads / m_stats.numLopReadCalls;
    
    ss << "usecsPerLopRead: " << usecsPerLopRead << ", ";

    double usecsPerMetric = 0.0;
    if(m_stats.totalMetricsRead)
        usecsPerMetric = m_stats.totalUsecInLopReads / m_stats.totalMetricsRead;

    ss << "usecsPerMetric: " << usecsPerMetric << ", ";

    std::string s = ss.str();
    if(logToStdout)
        printf("%s\n", s.c_str());
    if(logToDebugLog)
        PRINT_INFO("%s", "%s", s.c_str());
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ValidateGroupIdAndGetGpuIds(dcgmGpuGrp_t grpId, 
                                      unsigned int *groupId,
                                      std::vector<unsigned int> &gpuIds)
{
    *groupId = (unsigned int)(uintptr_t)grpId;
    dcgmReturn_t dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", *groupId);
        return dcgmReturn;
    }

    /* All GPUs in the group must be the same SKU */
    int areAllTheSameSku = 0;
    dcgmReturn = mpGroupManager->AreAllTheSameSku(0, *groupId, &areAllTheSameSku);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d %u", "Error %d from AreAllTheSameSku of groupId %u", 
                    dcgmReturn, *groupId);
        return dcgmReturn;
    }
    if(!areAllTheSameSku)
    {
        PRINT_ERROR("%u", "gpus of groupId %u are not all the same SKU", 
                    *groupId);
        return DCGM_ST_GROUP_INCOMPATIBLE;
    }
    
    dcgmReturn = mpGroupManager->GetGroupGpuIds(0, *groupId, gpuIds);
    if (DCGM_ST_OK != dcgmReturn || gpuIds.size() < 1)
    {
        PRINT_ERROR("%d %u", "Error %d from GetGroupGpuIds of groupId %u", 
                    dcgmReturn, *groupId);
        return dcgmReturn;
    }

    /* Only Volta is supported in LOP so far */
    if(m_gpus[gpuIds[0]].arch != LWML_CHIP_ARCH_VOLTA && m_gpus[gpuIds[0]].arch != LWML_CHIP_ARCH_TURING)
    {
        PRINT_DEBUG("%u %u", "gpuId %u chip arch %u is not VOLTA or TURING.", 
                    gpuIds[0], m_gpus[gpuIds[0]].arch);
        return DCGM_ST_PROFILING_NOT_SUPPORTED;
    }

    if(m_bypassSkuCheck)
    {
        PRINT_DEBUG("", "Ignoring SKU check");
    }
    else if(m_gpus[gpuIds[0]].brand != DCGM_GPU_BRAND_TESLA && m_gpus[gpuIds[0]].brand != DCGM_GPU_BRAND_QUADRO)
    {
        PRINT_DEBUG("%u %u", "gpuId %u chip brand %u is not Tesla or Lwdqro.", 
                    gpuIds[0], m_gpus[gpuIds[0]].brand);
        return DCGM_ST_PROFILING_NOT_SUPPORTED;
    }
    else if(m_gpus[gpuIds[0]].brand == DCGM_GPU_BRAND_QUADRO && m_gpus[gpuIds[0]].arch != LWML_CHIP_ARCH_TURING)
    {
        PRINT_DEBUG("%u %u", "gpuId %u chip brand %u is Lwdqro but not Turing or newer", 
                    gpuIds[0], m_gpus[gpuIds[0]].brand);
        return DCGM_ST_PROFILING_NOT_SUPPORTED;
    }

    /* Make sure that all GPUs in this group are ready to profile */
    std::vector<unsigned int>::iterator gpuIdIter;
    for(gpuIdIter = gpuIds.begin(); gpuIdIter != gpuIds.end(); ++gpuIdIter)
    {
        unsigned int gpuId = *gpuIdIter;
        
        if(m_gpus[gpuId].state != DcgmProfStateInitialized)
        {
            PRINT_WARNING("%u %u", "gpuId %u in state %u", gpuId, m_gpus[gpuId].state);
            return DCGM_ST_PROFILING_LIBRARY_ERROR;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmModuleProfiling::HelperAddFieldsToMetricGroup(dcgm_module_prof_gpu_t *gpu, 
                                                       dcgmProfGetMetricGroups_t *mg, 
                                                       unsigned short majorId, unsigned short minorId,
                                                       std::vector<unsigned short> &fieldIds)
{
    bool anyAdded = false;
    unsigned int i;
    dcgmProfMetricGroupInfo_t *mgi = &mg->metricGroups[mg->numMetricGroups];

    if(mg->numMetricGroups >= DCGM_PROF_MAX_NUM_GROUPS)
    {
        PRINT_ERROR("", "Unable to add another metric group. Increase DCGM_PROF_MAX_NUM_GROUPS");
        return false;
    }

    mgi->numFieldIds = 0;

    for(i = 0; i < fieldIds.size(); i++)
    {
        std::map<unsigned int, dcgmProfMetricIndex_t>::iterator metricIt = gpu->fieldIdsToMetrics.find(fieldIds[i]);
        if(metricIt == gpu->fieldIdsToMetrics.end())
            continue; /* Not supported for this GPU */
        
        /* Cache the major ID and minor ID for each metric */
        gpu->lopMetrics[metricIt->second].mgMajorId = majorId;
        gpu->lopMetrics[metricIt->second].mgMinorId = minorId;
        
        mgi->fieldIds[mgi->numFieldIds] = fieldIds[i];
        mgi->numFieldIds++;
        anyAdded = true;
    }

    if(anyAdded)
    {
        mgi->majorId = majorId;
        mgi->minorId = minorId;
        mg->numMetricGroups++;
    }

    return anyAdded;
}

/*****************************************************************************/
void DcgmModuleProfiling::BuildMetricGroupsGV100(dcgm_module_prof_gpu_t *gpu)
{
    if(gpu->fieldIdsToMetrics.size() < 1)
    {
        PRINT_ERROR("", "BuildBaseMetricListGV100 called before fieldIdsToEvents was built?");
        return;
    }

    dcgmProfGetMetricGroups_t *mg = &gpu->metricGroups; /* Shortlwt pointer */

    memset(mg, 0, sizeof(*mg));
    mg->version = dcgmProfGetMetricGroups_version;
    std::vector<unsigned short>fieldIds;

    gpu->smMajorGroup = 0;

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_SM_ACTIVE);
    fieldIds.push_back(DCGM_FI_PROF_SM_OCLWPANCY);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 1, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 2, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP64_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 3, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP32_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 4, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP16_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 5, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_DRAM_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 1, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PCIE_TX_BYTES);
    fieldIds.push_back(DCGM_FI_PROF_PCIE_RX_BYTES);
    HelperAddFieldsToMetricGroup(gpu, mg, 2, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_GR_ENGINE_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 3, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_LWLINK_TX_BYTES);
    fieldIds.push_back(DCGM_FI_PROF_LWLINK_RX_BYTES);
    HelperAddFieldsToMetricGroup(gpu, mg, 4, 0, fieldIds);
}

/*****************************************************************************/
void DcgmModuleProfiling::BuildMetricGroupsTU10x(dcgm_module_prof_gpu_t *gpu)
{
    if(gpu->fieldIdsToMetrics.size() < 1)
    {
        PRINT_ERROR("", "BuildBaseMetricListTU10x called before fieldIdsToEvents was built?");
        return;
    }

    dcgmProfGetMetricGroups_t *mg = &gpu->metricGroups; /* Shortlwt pointer */

    memset(mg, 0, sizeof(*mg));
    mg->version = dcgmProfGetMetricGroups_version;
    std::vector<unsigned short>fieldIds;

    gpu->smMajorGroup = 0;

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_SM_ACTIVE);
    fieldIds.push_back(DCGM_FI_PROF_SM_OCLWPANCY);
    fieldIds.push_back(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE);
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP32_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 1, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP64_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 2, fieldIds);    

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PIPE_FP16_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 0, 3, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_DRAM_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 1, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_PCIE_TX_BYTES);
    fieldIds.push_back(DCGM_FI_PROF_PCIE_RX_BYTES);
    HelperAddFieldsToMetricGroup(gpu, mg, 2, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_GR_ENGINE_ACTIVE);
    HelperAddFieldsToMetricGroup(gpu, mg, 3, 0, fieldIds);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_PROF_LWLINK_TX_BYTES);
    fieldIds.push_back(DCGM_FI_PROF_LWLINK_RX_BYTES);
    HelperAddFieldsToMetricGroup(gpu, mg, 4, 0, fieldIds);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessGetMetricGroups(dcgm_profiling_msg_get_mgs_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_profiling_msg_get_mgs_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    /* Verify group id is valid and get its GPU IDs */
    std::vector<unsigned int>gpuIds;
    unsigned int groupId = 0;

    dcgmReturn = ValidateGroupIdAndGetGpuIds(msg->metricGroups.groupId, &groupId, gpuIds);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn; /* Logging is handled by the helper */
    
    if(gpuIds.size() < 1)
    {
        PRINT_WARNING("%u", "Unexpected empty gpu groupId %u", groupId);
        return DCGM_ST_GROUP_IS_EMPTY;
    }

    /* Use the metric groups of the first GPU in the group */
    memcpy(&msg->metricGroups, &m_gpus[gpuIds[0]].metricGroups, sizeof(msg->metricGroups));
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmModuleProfiling::ClearLopWatchesForGpu(dcgm_module_prof_gpu_t *gpu)
{
    unsigned int i;

    /* Clear counters */
    for(i = 0; i < dcgmProfMetricCount; i++)
    {
        gpu->lopMetrics[i].values.clear();
    }

    gpu->eventQueryIntervalUsec = (long long)3600 * 1000000; /* Default = sleep for an hour */
    gpu->lastEventQueryUsec = 0;
    gpu->lastCacheUsec = 0;

    if(!gpu->lopGpu)
        return;
    
    gpu->lopGpu->DisableMetrics();
}

/*****************************************************************************/
void DcgmModuleProfiling::UpdateMemoryClocksAndBw(void)
{
    dcgmReturn_t dcgmReturn;

    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned short>fieldIds;

    fieldIds.push_back(DCGM_FI_DEV_MAX_MEM_CLOCK);

    dcgmGroupEntityPair_t ins;
    ins.entityGroupId = DCGM_FE_GPU;

    for(unsigned int i = 0; i < m_gpus.size(); i++)
    {
        ins.entityId = m_gpus[i].gpuId;
        entities.push_back(ins);
    }

    DcgmFvBuffer fvBuffer(FVBUFFER_GUESS_INITIAL_CAPACITY(m_gpus.size(), 1));

    dcgmReturn = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %d", "Unexpected st %d from GetMultipleLatestLiveSamples of %d entities.",
                    dcgmReturn, (int)entities.size());
    }

    dcgmBufferedFvLwrsor_t cursor = 0;
    for(dcgmBufferedFv_t *fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        dcgm_module_prof_gpu_t *gpu = &m_gpus[fv->entityId];

        if(fv->status != DCGM_ST_OK)
            gpu->memClockMhz = (double)0.0;
        else
            gpu->memClockMhz = (double)fv->value.i64;
        
        /* memClockMhz * 1000000 bytes per MiB * 2 copies per cycle. 4096 bit width of HBM /  8 bits per byte */
        gpu->maximumMemBandwidth = gpu->memClockMhz * 1000000.0 * 1024.0;

        PRINT_DEBUG("%f %u %d %f", "Got mem clock %f for gpuId %u. fv->status %d, maxBandwidth %f", 
                    gpu->memClockMhz, fv->entityId, (int)fv->status,
                    gpu->maximumMemBandwidth);

        /* The clock rate of V100 is hardcoded to 850 mhz in LOP. If our maximum memory clock is not 850, we need to adjust
           our bandwidth for the difference in clock */
        double lopMemClock = 850.0;
        if(gpu->arch == LWML_CHIP_ARCH_TURING)
            lopMemClock = 7000.0; /* Turing is hardcoded to 7000 mhz */

        if(gpu->memClockMhz != 0.0 && gpu->memClockMhz != lopMemClock)
        {
            /* Since we're adjusting the divisor for our values, multiply it by the ilwerse of the clock rates */
            double adjustment = gpu->memClockMhz / lopMemClock;
            gpu->lopMetrics[dcgmProfMetricDramActive].metricDivisor *= adjustment;
            PRINT_DEBUG("%u %.3f %.3f", "gpuId %u adjustment %.3f, metricDivisor %.3f", 
                        gpu->gpuId, adjustment, gpu->lopMetrics[dcgmProfMetricDramActive].metricDivisor);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ColwertFieldIdsToMetricIds(dcgm_module_prof_gpu_t *gpu,
                                                             unsigned short *fieldIds, int numFieldIds,
                                                             std::vector<dcgmProfMetricIndex_t> &metricIds)
{
    metricIds.clear();
    metricIds.reserve(numFieldIds);
    std::map<unsigned int, dcgmProfMetricIndex_t>::const_iterator it;    

    for(int i = 0; i < numFieldIds; i++)
    {
        /* Is this field ID supported for this GPU? */
        it = gpu->fieldIdsToMetrics.find(fieldIds[i]);
        if(it == gpu->fieldIdsToMetrics.end())
        {
            PRINT_ERROR("%u %u", "Field ID %u is not supported for GPU %u", fieldIds[i], gpu->gpuId);
            return DCGM_ST_NOT_SUPPORTED;
        }

        metricIds.push_back(it->second);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::BuildMetricListFromFieldIdList(
                                        dcgm_module_prof_gpu_t *gpu,
                                        std::vector<const char *> (&metricTags)[DLG_MAX_METRIC_GROUPS],
                                        std::vector<unsigned int> (&metricIds)[DLG_MAX_METRIC_GROUPS],
                                        dcgm_profiling_msg_watch_fields_t *msg, 
                                        int *numPassGroups)
{
    dcgmReturn_t dcgmReturn;
    *numPassGroups = 0;
    std::vector<dcgmProfMetricIndex_t> requestedMetrics;
    std::map<unsigned int, std::vector<dcgmProfMetricIndex_t> > swappedMetrics; /* Metrics that will be in a single 
                                                                                   config like Tensor act. The key is
                                                                                   the mgMajorId of the metric */
    std::vector<dcgmProfMetricIndex_t> notSwappedMetrics; /* Metrics that will be in every config like Gr Act */

    /* Check the validity of the requested fieldIds and colwert them into metric IDs */
    dcgmReturn = ColwertFieldIdsToMetricIds(gpu, msg->watchFields.fieldIds, 
                                            msg->watchFields.numFieldIds,
                                            requestedMetrics);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;
    
    /* See if we have multiple passes worth of fields */
    for(size_t i = 0; i < requestedMetrics.size(); i++)
    {
        dcgmProfMetricIndex_t mId = requestedMetrics[i];
        dcgmProfMetric_t *metric = &gpu->lopMetrics[mId];

        if(metric->mgMajorId != gpu->smMajorGroup)
        {
            notSwappedMetrics.push_back(mId);
            continue;
        }
        
        swappedMetrics[metric->mgMinorId].push_back(mId);    
    }

    PRINT_DEBUG("%zu %zu", "swappedMetrics.size() %zu, notSwappedMetrics.size() %zu", 
                swappedMetrics.size(), notSwappedMetrics.size());

    if(swappedMetrics.size() > DLG_MAX_METRIC_GROUPS)
    {
        PRINT_ERROR("%zu %d", "Got %zu metric groups. More than our limit of %d", 
                    swappedMetrics.size(), DLG_MAX_METRIC_GROUPS);
        return DCGM_ST_PROFILING_MULTI_PASS;
    }

    /* See if there are multiple passes worth of metrics in the requested metrics */
    bool haveMultiplePasses = swappedMetrics.size() > 1;

    if(swappedMetrics.size() < 2)
    {
        /* There's going to be at least one metric per fieldId */
        metricTags[0].reserve(requestedMetrics.size());
        metricIds[0].reserve(requestedMetrics.size());

        for(size_t i = 0; i < requestedMetrics.size(); i++)
        {
            metricTags[0].push_back(gpu->lopMetrics[requestedMetrics[i]].lopTag);
            metricIds[0].push_back(requestedMetrics[i]);
        }
        *numPassGroups = 1;
        return DCGM_ST_OK;
    }

    /* Walk each of swappedMetrics, starting a metric array with its entries, 
       followed by everything in notSwappedMetrics */
    std::map<unsigned int, std::vector<dcgmProfMetricIndex_t> >::iterator swappedIt;
    std::vector<dcgmProfMetricIndex_t>::iterator metricIt;
    int passGroupIndex = 0;

    for(swappedIt = swappedMetrics.begin(); swappedIt != swappedMetrics.end(); ++swappedIt)
    {
        PRINT_DEBUG("%d", "Pass group %d starting", passGroupIndex);

        for(metricIt = swappedIt->second.begin(); metricIt != swappedIt->second.end(); ++metricIt)
        {
            const char *metricTag = gpu->lopMetrics[*metricIt].lopTag;
            PRINT_DEBUG("%u %s %d", "Adding swappedMetrics metricId %u, tag %s to passGroup %d", *metricIt, metricTag, passGroupIndex);
            metricTags[passGroupIndex].push_back(metricTag);
            metricIds[passGroupIndex].push_back(*metricIt);
        }

        for(metricIt = notSwappedMetrics.begin(); metricIt != notSwappedMetrics.end(); ++metricIt)
        {
            const char *metricTag = gpu->lopMetrics[*metricIt].lopTag;
            PRINT_DEBUG("%u %s %d", "Adding notSwappedMetric metricId %u, tag %s to passGroup %d", *metricIt, metricTag, passGroupIndex);
            metricTags[passGroupIndex].push_back(metricTag);
            metricIds[passGroupIndex].push_back(*metricIt);
        }

        passGroupIndex++;
    }

    *numPassGroups = passGroupIndex;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessWatchFieldsGpu(dcgm_profiling_msg_watch_fields_t *msg,
                                                        dcgm_module_prof_gpu_t *gpu)
{
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;
    unsigned int i;
    
    gpu->enabledFieldIds.clear();

    /* Set the metric query interval to the watch interval */
    gpu->eventQueryIntervalUsec = msg->watchFields.updateFreq;

    /* Tags and corresponding IDs of the metrics we are going to watch */
    std::vector<const char *>metricTags[DLG_MAX_METRIC_GROUPS];
    std::vector<unsigned int>metricIds[DLG_MAX_METRIC_GROUPS];
    int numPassGroups = 0; /* How many passes are we doing to have to do to stay single-pass */

    dcgmReturn = BuildMetricListFromFieldIdList(gpu, metricTags, metricIds, msg, &numPassGroups);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn;

    dcgmReturn = gpu->lopGpu->InitializeWithMetrics(metricTags, metricIds, numPassGroups);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %d", "InitializeWithMetrics returned %d for %d metrics", 
                    dcgmReturn, (int)metricTags[0].size());
        return dcgmReturn;
    }

    if(numPassGroups < 1 || metricTags[0].size() < 1)
    {
        PRINT_WARNING("%u", "No lop metrics to watch for gpuId %u", gpu->gpuId);
        return DCGM_ST_NOT_SUPPORTED;
    }

    if(numPassGroups > 1)
    {
        /* If we have more than one pass worth of metrics, adjust the eventQueryIntervalUsec 
           so that we do statistical sampling. If our watch interval is less than a second,
           then wake up every updateInterval / numPassGroups.
           Otherwise, wake up every 1 second / numPassGroups.
           Note that we are setting our interval to 95% to allow for 5% overhead of callwlating the metrics.
           Otherwise, the last 1/n slice is unlikely to happen before we sample our metrics */
        if(msg->watchFields.updateFreq < 1000000)
            gpu->eventQueryIntervalUsec = (timelib64_t)((0.95 * msg->watchFields.updateFreq) / numPassGroups);
        else
            gpu->eventQueryIntervalUsec = 950000 / numPassGroups;
        PRINT_DEBUG("%lld %lld %d", "Adjusted eventQueryIntervalUsec to %lld for "
                    "updateFreq %lld, numPassgroups %d", (long long)gpu->eventQueryIntervalUsec,
                    (long long)msg->watchFields.updateFreq, numPassGroups);
    }

    /* Since there weren't any errors above, set all of the fields to watched */
    for(unsigned int i = 0; i < msg->watchFields.numFieldIds; i++)
        gpu->enabledFieldIds.push_back(msg->watchFields.fieldIds[i]);

    if(!m_profilingIsPaused)
    {
        PRINT_DEBUG("%u", "Enabling metrics for gpuId %u", gpu->gpuId);
        dcgmReturn = gpu->lopGpu->EnableMetrics();
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "EnableMetrics returned %d", dcgmReturn);
            return dcgmReturn;
        }
    }
    else
    {
        PRINT_DEBUG("%u", "Skipping EnableMetrics() for gpuId %u since profiling is paused.", gpu->gpuId);
    }

    /* Make sure watches are set in the cache manager for each of the fieldIds so
       the watches are tracked and have a quota policy. Note that we may be getting called from the host 
       engine once it has added watches for these fields. That's OK since the cache manager is smart enough
       to find the same watcher and simply update it. */
    
    DcgmWatcher watcher;
    watcher.watcherType = DcgmWatcherTypeClient;
    watcher.connectionId = msg->header.connectionId;

    std::vector<unsigned short>::const_iterator fieldIdIter;
    for(fieldIdIter = gpu->enabledFieldIds.begin(); fieldIdIter != gpu->enabledFieldIds.end(); ++fieldIdIter)
    {
        dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpu->gpuId, *fieldIdIter, 
                                                   msg->watchFields.updateFreq, 
                                                   msg->watchFields.maxKeepAge, 
                                                   msg->watchFields.maxKeepSamples, 
                                                   watcher, false);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u %u", "AddFieldWatch returned %d for gpuId %u, fieldId %u", 
                        dcgmReturn, gpu->gpuId, *fieldIdIter);
            return dcgmReturn;
        }
        
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessWatchFields(dcgm_profiling_msg_watch_fields_t *msg)
{
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;

    dcgmReturn = CheckVersion(&msg->header, dcgm_profiling_msg_watch_fields_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    if(!msg->watchFields.updateFreq)
    {
        PRINT_ERROR("", "Invalid 0 updateFreq");
        return DCGM_ST_BADPARAM;
    }
    if(!msg->watchFields.numFieldIds)
    {
        PRINT_ERROR("", "Invalid 0 numFieldIds");
        return DCGM_ST_BADPARAM;
    }

    /* Does another client already have watches established? */
    if(msg->header.connectionId != DCGM_CONNECTION_ID_NONE)
    {
        if(m_watcherConnId != DCGM_CONNECTION_ID_NONE && m_watcherConnId != msg->header.connectionId)
        {
            PRINT_ERROR("%u %u", "Unable to watch profiling metrics for clientId %u. Already watched by clientId %u",
                        msg->header.connectionId, m_watcherConnId);
            return DCGM_ST_IN_USE;
        }
    }
    
    /* Verify group id is valid and get its GPU IDs */
    std::vector<unsigned int>gpuIds;
    unsigned int groupId = 0;

    dcgmReturn = ValidateGroupIdAndGetGpuIds(msg->watchFields.groupId, &groupId, gpuIds);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn; /* Logging is handled by the helper */
    
    /* Clear any previous watches and counters */
    UnwatchAll();

    unsigned int i, j;
    for(i = 0; i < gpuIds.size() && retSt == DCGM_ST_OK; i++)
    {
        dcgm_module_prof_gpu_t *gpu = &m_gpus[gpuIds[i]];

        retSt = ProcessWatchFieldsGpu(msg, gpu);
        if(retSt != DCGM_ST_OK)
            break;
    }

    /* Did something fail above? If so, unwatch everything */
    if(retSt != DCGM_ST_OK)
        UnwatchAll();

    m_cacheIntervalUsec = msg->watchFields.updateFreq;
    m_watcherConnId = msg->header.connectionId;

    /* Make sure we have an initial record for all of the metrics we care about */
    ReadAndCacheMetrics();
    
    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessUnwatchFields(dcgm_profiling_msg_unwatch_fields_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int> gpuIds;

    dcgmReturn = CheckVersion(&msg->header, dcgm_profiling_msg_unwatch_fields_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    dcgmReturn = ValidateGroupIdAndGetGpuIds(msg->unwatchFields.groupId, &groupId, gpuIds);
    if(dcgmReturn != DCGM_ST_OK)
        return dcgmReturn; /* Logging is handled by the helper */

    if(msg->header.connectionId != m_watcherConnId)
    {
        PRINT_ERROR("%u %u", "Got ProcessUnwatchFields from connectionId %u. Actually owned by user %u", 
                    msg->header.connectionId, m_watcherConnId);
        return DCGM_ST_IN_USE;
    }
    
    std::vector<unsigned short>::const_iterator fieldIdIter;
    std::vector<unsigned int>::const_iterator gpuIdIter;
    DcgmWatcher watcher;
    watcher.watcherType = DcgmWatcherTypeClient;
    watcher.connectionId = msg->header.connectionId;
    
    /* Unwatch every field in the cache manager for this client */
    for(gpuIdIter = gpuIds.begin(); gpuIdIter != gpuIds.end(); ++gpuIdIter)
    {
        for(fieldIdIter = m_gpus[*gpuIdIter].enabledFieldIds.begin(); 
            fieldIdIter != m_gpus[*gpuIdIter].enabledFieldIds.end(); ++fieldIdIter)
        {
            dcgmReturn = mpCacheManager->RemoveFieldWatch(DCGM_FE_GPU, *gpuIdIter, *fieldIdIter, 
                                                          1, watcher);
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u %u", "RemoveFieldWatch returned %d for gpuId %u, fieldId %u", 
                            dcgmReturn, *gpuIdIter, *fieldIdIter);
                return dcgmReturn;
            }
        }
    }

    /* Clear everything from this connection for now */
    UnwatchAll();

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmModuleProfiling::OnGroupRemove(unsigned int groupId)
{
}

/*****************************************************************************/
void DcgmModuleProfiling::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
    /* Todo: Move this to LwcmHostEngineHandler and remove this virtual function
              once we have multiprocess modules */
    dcgm_core_msg_client_disconnect_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length = sizeof(msg);
    msg.header.version = dcgm_core_msg_client_disconnect_version;
    msg.header.moduleId = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_CLIENT_DISCONNECT;
    msg.header.connectionId = connectionId; /* Redundant but explicit */
    msg.connectionId = connectionId;
    ProcessMessage((dcgm_module_command_header_t *)&msg);
}

/*****************************************************************************/
void DcgmModuleProfiling::TryAddingFieldIdWithMetric(dcgm_module_prof_gpu_t *gpu,
                                                     unsigned int fieldId,
                                                     dcgmProfMetricIndex_t metricIndex)
{
    if(!gpu || !fieldId)
    {
        PRINT_ERROR("", "Bad parameter");
        return;
    }

    if(gpu->fieldIdsToMetrics.find(fieldId) != gpu->fieldIdsToMetrics.end())
    {
        PRINT_ERROR("%u %u", "gpu %u already has a fieldIdsToMetrics entry for fieldId %u",
                    gpu->gpuId, fieldId);
        return;
    }

    /* See if this metric is supported. If it is, skip adding this fieldId */
    if(!gpu->lopMetrics[metricIndex].isSupported)
    {
        PRINT_DEBUG("%u %u %u", "FieldId %u is not supported for gpuId %u due to metricIndex %u",
                    fieldId, gpu->gpuId, metricIndex);
        return;
    }

    PRINT_INFO("%u %u %u", "fieldId %u is supported for gpuId %u as metricIndex %u",
               fieldId, gpu->gpuId, metricIndex);

    /* Add the fieldId mapping to its metric index */
    gpu->fieldIdsToMetrics[fieldId] = metricIndex;
}

/*****************************************************************************/
void DcgmModuleProfiling::BuildLopMetricList(unsigned int gpuId)
{
    unsigned int i;
    dcgm_module_prof_gpu_t *gpu = &m_gpus[gpuId];

    for(i = 0; i < dcgmProfMetricCount; i++)
    {
        gpu->lopMetrics[i].id = (dcgmProfMetricIndex_t)i;
        gpu->lopMetrics[i].isSupported = false;
        gpu->lopMetrics[i].metricDivisor = 1.0;
        gpu->lopMetrics[i].minAllowedValue = 0.0; /* All metrics are positive for now */
        gpu->lopMetrics[i].maxAllowedValue = 1.0; /* Default to ratio maximum of 1.0. Bandwidth values will
                                                     be the exception */
    }

    /* Set the LOP-understood tags that we'll use to identify these metrics to LOP */
    gpu->lopMetrics[dcgmProfMetricGrActive].lopTag = "gr__cycles_active.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricGrActive].metricDivisor = 100.0;

    gpu->lopMetrics[dcgmProfMetricSmActive].lopTag = "sm__cycles_active.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricSmActive].metricDivisor = 100.0;

    if(gpu->arch == LWML_CHIP_ARCH_VOLTA)
        gpu->lopMetrics[dcgmProfMetricSmOclwpancy].lopTag = "sm__warps_active.avg.pct_of_peak_sustained_elapsed";
    else
        gpu->lopMetrics[dcgmProfMetricSmOclwpancy].lopTag = "sm__warps_active_realtime.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricSmOclwpancy].metricDivisor = 100.0;

    if(gpu->arch == LWML_CHIP_ARCH_VOLTA)
        gpu->lopMetrics[dcgmProfMetricPipeTensorActive].lopTag = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed";
    else /* Turing or newer */
        gpu->lopMetrics[dcgmProfMetricPipeTensorActive].lopTag = "sm__pipe_shared_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricPipeTensorActive].metricDivisor = 100.0;

    if(gpu->arch == LWML_CHIP_ARCH_VOLTA)
        gpu->lopMetrics[dcgmProfMetricDramActive].lopTag = "dramc__throughput.avg.pct_of_peak_sustained_elapsed";
    else /* Turing or newer */
        gpu->lopMetrics[dcgmProfMetricDramActive].lopTag = "dramc__throughput_realtime.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricDramActive].metricDivisor = 100.0;
    /* Note: This divisor is adjusted by UpdateMemoryClocksAndBw() */

    gpu->lopMetrics[dcgmProfMetricPipeFp64Active].lopTag = "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricPipeFp64Active].metricDivisor = 100.0;

    if(gpu->arch == LWML_CHIP_ARCH_VOLTA)
    {
        gpu->lopMetrics[dcgmProfMetricPipeFp32Active].lopTag = "smsp__inst_exelwted_pipe_fma.avg.pct_of_peak_sustained_elapsed";
        gpu->lopMetrics[dcgmProfMetricPipeFp32Active].metricDivisor = 50.0; /* x2 for SMSP, divide by 100 for pct -> ratio */
    }
    else /* Turing or newer */
    {
        gpu->lopMetrics[dcgmProfMetricPipeFp32Active].lopTag = "sm__inst_exelwted_pipe_fma_realtime.avg.pct_of_peak_sustained_elapsed";
        gpu->lopMetrics[dcgmProfMetricPipeFp32Active].metricDivisor = 100.0;
    }

    gpu->lopMetrics[dcgmProfMetricPipeFp16Active].lopTag = "sm__inst_exelwted_pipe_fp16.avg.pct_of_peak_sustained_elapsed";
    gpu->lopMetrics[dcgmProfMetricPipeFp16Active].metricDivisor = 100.0;

    /* PCI-e bandwidth is from here https://en.wikipedia.org/wiki/PCI_Express#History_and_revisions. 
       Using 16 GB since x16 3.0 maxes out at 15.75 GB/sec */
    gpu->lopMetrics[dcgmProfMetricPcieTxBytes].lopTag = "pcie__write_bytes.sum.per_second";
    gpu->lopMetrics[dcgmProfMetricPcieTxBytes].maxAllowedValue = (double)16 * 1024 * 1024 * 1024;

    gpu->lopMetrics[dcgmProfMetricPcieRxBytes].lopTag = "pcie__read_bytes.sum.per_second";
    gpu->lopMetrics[dcgmProfMetricPcieRxBytes].maxAllowedValue = (double)16 * 1024 * 1024 * 1024;

    /* LwLink metrics - the following are the user data versions (don't include protocol overhead) 
       LwLink bandwidth max is for V100. 6 LwLinks x 25 GB/s = 150 GB/s */
    gpu->lopMetrics[dcgmProfMetricLwLinkTxBytes].lopTag = "lwltx__bytes_src_user.sum.per_second";
    gpu->lopMetrics[dcgmProfMetricLwLinkTxBytes].maxAllowedValue = (double)150 * 1024 * 1024 * 1024;
    gpu->lopMetrics[dcgmProfMetricLwLinkRxBytes].lopTag = "lwlrx__bytes_src_user.sum.per_second";
    gpu->lopMetrics[dcgmProfMetricLwLinkRxBytes].maxAllowedValue = (double)150 * 1024 * 1024 * 1024;

    /* Get the ID of each event. Not being able to get the ID means it's not available for this GPU */
    for(i = 0; i < dcgmProfMetricCount; i++)
    {
        if(!gpu->lopMetrics[i].lopTag || !gpu->lopMetrics[i].lopTag[0])
        {
            PRINT_DEBUG("%u %u", "gpuId %u, id %u, lopTag NULL is not supported.",
                        gpuId, gpu->lopMetrics[i].id);
            gpu->lopMetrics[i].isSupported = false;
            continue;
        }

        bool isValid = gpu->lopGpu->IsMetricNameValid(gpu->lopMetrics[i].lopTag);
        if(!isValid)
        {
            PRINT_DEBUG("%u %u %s", "gpuId %u, id %u, lopTag %s was not valid and is thus not supported.",
                        gpuId, gpu->lopMetrics[i].id, gpu->lopMetrics[i].lopTag);
            gpu->lopMetrics[i].isSupported = false;
            continue;
        }

        PRINT_DEBUG("%u %u %s", "gpuId %u, id %u, lopTag %s is supported.",
                    gpuId, gpu->lopMetrics[i].id, gpu->lopMetrics[i].lopTag);
        gpu->lopMetrics[i].isSupported = true;
    }
    

    /* Add each field ID and its corresponding metric to this GPU. If the metric is not
       supported, this will be ignored */

    /* DCGM_FI_PROF_GR_ENGINE_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_GR_ENGINE_ACTIVE, dcgmProfMetricGrActive);

    /* DCGM_FI_PROF_SM_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_SM_ACTIVE, dcgmProfMetricSmActive);

    /* DCGM_FI_PROF_SM_OCLWPANCY */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_SM_OCLWPANCY, dcgmProfMetricSmOclwpancy);

    /* DCGM_FI_PROF_PIPE_TENSOR_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, dcgmProfMetricPipeTensorActive);

    /* DCGM_FI_PROF_DRAM_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_DRAM_ACTIVE, dcgmProfMetricDramActive);

    /* DCGM_FI_PROF_PIPE_FP64_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PIPE_FP64_ACTIVE, dcgmProfMetricPipeFp64Active);

    /* DCGM_FI_PROF_PIPE_FP32_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PIPE_FP32_ACTIVE, dcgmProfMetricPipeFp32Active);

    /* DCGM_FI_PROF_PIPE_FP16_ACTIVE */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PIPE_FP16_ACTIVE, dcgmProfMetricPipeFp16Active);

    /* DCGM_FI_PROF_PCIE_TX_BYTES */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PCIE_TX_BYTES, dcgmProfMetricPcieTxBytes);

    /* DCGM_FI_PROF_PCIE_RX_BYTES */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_PCIE_RX_BYTES, dcgmProfMetricPcieRxBytes);

    /* DCGM_FI_PROF_LWLINK_TX_BYTES */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_LWLINK_TX_BYTES, dcgmProfMetricLwLinkTxBytes);

    /* DCGM_FI_PROF_LWLINK_RX_BYTES */
    TryAddingFieldIdWithMetric(gpu, DCGM_FI_PROF_LWLINK_RX_BYTES, dcgmProfMetricLwLinkRxBytes);
}

/*****************************************************************************/
int GetLopGpuId(LWPW_Device_GetPciBusIds_Params const &getBusIds, dcgmcm_gpu_info_cached_t const gpuInfo)
{
    for (size_t i = 0; i < getBusIds.numDevices; i++)
    {
        if ((getBusIds.pBusIds[i].domain == gpuInfo.pciInfo.domain) &&
            (getBusIds.pBusIds[i].device == gpuInfo.pciInfo.device) &&
            (getBusIds.pBusIds[i].bus == gpuInfo.pciInfo.bus))
        {
            return i;
        }
    }

    // Did not find a matching LopGpuId device
    return -1;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::InitLop(void)
{
    dcgmReturn_t dcgmReturn;

    /* Find out info about every GPU in the system so we can correllate them to
       lwca-visible GPUs */
    std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
    dcgmReturn = mpCacheManager->GetAllGpuInfo(gpuInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetAllGpuInfo", (int)dcgmReturn);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Prepare m_gpus to get GPUs */
    m_gpus.resize(gpuInfo.size());

    /* Initialize the LOP target and host */
    LWPA_Status lwpaStatus = LWPA_InitializeTarget();
    if(lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "LWPA_InitializeTarget returned %d", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    PRINT_DEBUG("", "LWPA_InitializeTarget() was successful.");

    LWPW_LWDA_LoadDriver_Params loadDriverParams;
    memset(&loadDriverParams, 0, sizeof(loadDriverParams));
    loadDriverParams.structSize = LWPW_LWDA_LoadDriver_Params_STRUCT_SIZE;
    lwpaStatus = LWPW_LWDA_LoadDriver(&loadDriverParams);
    if(lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "LWPW_LWDA_LoadDriver returned %d", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    PRINT_DEBUG("", "LWPW_LWDA_LoadDriver() was successful.");

    lwpaStatus = LWPA_InitializeHost();
    if(lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "LWPA_InitializeHost returned %d", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    PRINT_DEBUG("", "LWPA_InitializeHost() was successful.");

    LWPW_PciBusId busIds[DCGM_MAX_NUM_DEVICES];
    LWPW_Device_GetPciBusIds_Params getBusIds;
    getBusIds.structSize = sizeof(getBusIds);
    getBusIds.pPriv = NULL;
    getBusIds.pBusIds = new LWPW_PciBusId[gpuInfo.size()];
    size_t numDevices = 0;
    lwpaStatus = LWPA_GetDeviceCount(&numDevices);
    if (lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "Unable to get a device count: LWPA_GetDeviceCount returned error %d", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }
    getBusIds.numDevices = numDevices;
    lwpaStatus = LWPW_Device_GetPciBusIds(&getBusIds);
    if (lwpaStatus != LWPA_STATUS_SUCCESS)
    {
        PRINT_ERROR("%d", "Couldn't get PCIe information from LWPW_Device_GetPciBusIds: %d", lwpaStatus);
        return DCGM_ST_PROFILING_LIBRARY_ERROR;
    }

    /* Initialize LOP for each GPU */
    for(unsigned int i = 0; i < gpuInfo.size(); i++)
    {
        unsigned int gpuId = gpuInfo[i].gpuId;
        dcgm_module_prof_gpu_t *gpu;

        gpu = &m_gpus[gpuId];

        gpu->gpuId = gpuId;
        gpu->state = DcgmProfStateUninitialized;
        gpu->arch = gpuInfo[i].arch;
        gpu->brand = gpuInfo[i].brand;
        gpu->memClockMhz = 0.0;
        gpu->maximumMemBandwidth = 0.0;
        gpu->lastEventQueryUsec = 0;
        gpu->eventQueryIntervalUsec = 0;
        gpu->lastCacheUsec = 0;
        int lopGpuId = GetLopGpuId(getBusIds, gpuInfo[i]);
        if (lopGpuId == -1)
        {
            PRINT_ERROR("%u %x %x", "Could not find any LOP Gpu with device %u, domain %x, bus %x",
                        gpuInfo[i].pciInfo.device, gpuInfo[i].pciInfo.domain, gpuInfo[i].pciInfo.bus);
            return DCGM_ST_GENERIC_ERROR;
        }

        gpu->lopGpu = new DcgmLopGpu(lopGpuId);

        dcgmReturn = gpu->lopGpu->Init();
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "lopGpu->Init() failed with %d", dcgmReturn);
            return dcgmReturn;
        }

        /* Get a list of events that are supported for each GPU */
        BuildLopMetricList(gpuId);

        /* Build our metric group get response based on which fields and events are supported */
        if(gpu->arch == LWML_CHIP_ARCH_VOLTA)
            BuildMetricGroupsGV100(gpu);
        else
            BuildMetricGroupsTU10x(gpu);

        PRINT_INFO("%u", "LOP monitoring initialized successfully for gpuId %u.", gpuId);
        gpu->state = DcgmProfStateInitialized;
    }

    /* Get the memory clocks. When we support GPUs with multiple p-states, we will have to update this as well */
    UpdateMemoryClocksAndBw();

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmModuleProfiling::ReadGpuMetrics(dcgm_module_prof_gpu_t *gpu, 
                                         unsigned int *gpuWakeupMs,
                                         timelib64_t *now)
{
    unsigned int i, j;
    dcgmReturn_t dcgmReturn;

    /* Are any field IDs enabled for this GPU? */
    if(gpu->enabledFieldIds.size() < 1)
        return;

    /* Wake up when we need to query another value from LOP */
    if(!gpu->lastEventQueryUsec)
        *gpuWakeupMs = gpu->eventQueryIntervalUsec / 1000;
    else
    {
        long long gpuWakeupUsec = gpu->eventQueryIntervalUsec - (*now - gpu->lastEventQueryUsec);
        if(gpuWakeupUsec <= 0)
            *gpuWakeupMs = gpu->eventQueryIntervalUsec / 1000;
        else
            *gpuWakeupMs = (unsigned int)(gpuWakeupUsec / 1000);
    }

    //printf("gpuWakeupMs %u, eventQueryIntervalUsec %lld, now %lld, lastEventQueryUsec %lld\n",
    //       *gpuWakeupMs, (long long)gpu->eventQueryIntervalUsec, (long long)*now, (long long)gpu->lastEventQueryUsec);

    if(*now - gpu->lastEventQueryUsec < gpu->eventQueryIntervalUsec)
        return; /* Nothing to do */
    gpu->lastEventQueryUsec = *now; /* Set this before we do expensive queries so we don't get data gaps of the size
                                       of how long lopGpu->GetSamples() took */

    size_t numCountersRead = 0;
    
    /* Sample the events */
    timelib64_t beforeTs = *now;
    
    dcgmReturn = gpu->lopGpu->GetSamples(gpu->sampleBuffer);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got error %d from GetSamples()", dcgmReturn);
        return;
    }

    if(gpu->sampleBuffer.size() < 1)
    {
        PRINT_ERROR("", "No samples returned from lop. Hmmmm.");
        return;
    }

    for(i = 0; i < gpu->sampleBuffer.size(); i++)
    {
        unsigned int metricId = gpu->sampleBuffer[i].metricId;
        double value = gpu->sampleBuffer[i].value;

        if(!DCGM_FP64_IS_BLANK(value))
        { 
            if(gpu->lopMetrics[metricId].metricDivisor != 1.0 && gpu->lopMetrics[metricId].metricDivisor != 0.0)
            {
                value /= gpu->lopMetrics[metricId].metricDivisor;
            }

            if(value < gpu->lopMetrics[metricId].minAllowedValue || value > gpu->lopMetrics[metricId].maxAllowedValue)
            {
                PRINT_ERROR("%u %u %f %f %f", "Ignoring gpuId %u, metricId %u with value %f outside of %f <= X <= %f",
                            gpu->gpuId, metricId, value, gpu->lopMetrics[metricId].minAllowedValue, 
                            gpu->lopMetrics[metricId].maxAllowedValue);
                value = DCGM_FP64_BLANK;
            }
        }

        gpu->lopMetrics[metricId].values.push_back(value);
    }

    m_stats.totalMetricsRead += gpu->sampleBuffer.size();
    *now = timelib_usecSince1970();
    m_stats.totalUsecInLopReads += *now - beforeTs;
    m_stats.numLopReadCalls++;
}

/*****************************************************************************/
void DcgmModuleProfiling::CacheAvgInt64BandwidthValue(dcgm_module_prof_gpu_t *gpu, 
                                                      unsigned short fieldId,
                                                      dcgmProfMetricIndex_t metricIndex, 
                                                      timelib64_t *now,
                                                      DcgmFvBuffer *fvBuffer)
{
    dcgmProfMetric_t *lopMetric = &gpu->lopMetrics[metricIndex];
    long long value = DCGM_INT64_BLANK;
    /* Note: we are no longer dividing by timeDiff here since the metrics we use
             for bandwidth are .per_second already */

    if(!lopMetric->isSupported)
    {
        PRINT_DEBUG("%u", "Field ID %u is being cached with a missing metric.", fieldId);
        value = DCGM_INT64_BLANK; /* Redudant but colweys intent */
    }
    else if(m_profilingIsPaused)
    {
        PRINT_DEBUG("%u", "Profiling is paused. Writing blank for fieldId %u", fieldId);
        value = DCGM_INT64_BLANK; /* Redudant but colweys intent */
    }
    else if(lopMetric->values.size() < 1)
    {
        PRINT_DEBUG("%u", "Field ID %u is being cached with no samples available", fieldId);
        value = 0;
    }
    else
    {
        for(int i = 0; i < (int)lopMetric->values.size(); i++)
        {
            double v = lopMetric->values[i];
            if(!DCGM_FP64_IS_BLANK(v))
            {
                if(DCGM_INT64_IS_BLANK(value))
                    value = (long long)v;
                else
                    value += (long long)v;
            }
        }

        /* The LOP PCIe/LwLink bandwidth metrics end in .per_second, which means they are already 
           divided by the time difference. We need to average them rather than just summing them.
           https://jirasw.lwpu.com/browse/DCGM-1362 */
        if(!DCGM_INT64_IS_BLANK(value))
            value /= (long long)lopMetric->values.size();
    }

    //printf("fieldId %u, gpu %u, val %lld, ts %u\n", 
    //       fieldId, gpu->gpuId, sample.val.i64, (unsigned int)(sample.timestamp/1000000));
    fvBuffer->AddInt64Value(DCGM_FE_GPU, gpu->gpuId, fieldId,
                            value, *now, DCGM_ST_OK);
}

/*****************************************************************************/
void DcgmModuleProfiling::CacheAvgDoubleValue(dcgm_module_prof_gpu_t *gpu, 
                                              unsigned short fieldId,
                                              dcgmProfMetricIndex_t metricIndex, 
                                              timelib64_t *now,
                                              DcgmFvBuffer *fvBuffer)
{
    dcgmProfMetric_t *lopMetric = &gpu->lopMetrics[metricIndex];
    double value = DCGM_FP64_BLANK;

    if(!lopMetric->isSupported)
    {
        PRINT_DEBUG("%u", "Field ID %u is being cached with a missing metric.", fieldId);
        value = DCGM_FP64_BLANK;
    }
    else if(m_profilingIsPaused)
    {
        PRINT_DEBUG("%u", "Profiling is paused. Writing blank for fieldId %u", fieldId);
        value = DCGM_FP64_BLANK; /* Redudant but colweys intent */
    }
    else if(lopMetric->values.size() < 1)
    {
        PRINT_DEBUG("%u", "Field ID %u is being cached with no samples available", fieldId);
        value = 0.0;
    }
    else
    {
        /* Average raw input values value */
        for(int i = 0; i < (int)lopMetric->values.size(); i++)
        {
            double v = lopMetric->values[i];
            if(!DCGM_FP64_IS_BLANK(v))
            {
                if(DCGM_FP64_IS_BLANK(value))
                    value = v;
                else
                    value += v;
            }
        }

        if(!DCGM_FP64_IS_BLANK(value))
            value /= (double)lopMetric->values.size();
    }

    //printf("fieldId %u, gpu %u, val %f, ts %u\n", 
    //       fieldId, gpu->gpuId, sample.val.dbl, (unsigned int)(sample.timestamp/1000000));
    fvBuffer->AddDoubleValue(DCGM_FE_GPU, gpu->gpuId, fieldId,
                             value, *now, DCGM_ST_OK);
}

/*****************************************************************************/
void DcgmModuleProfiling::SendGpuMetricsToCache(dcgm_module_prof_gpu_t *gpu, 
                                                unsigned int *gpuWakeupMs,
                                                timelib64_t *now,
                                                DcgmFvBuffer *fvBuffer)
{
    if(!gpu->enabledFieldIds.size())
        return; /* No fields enabled. Nothing to do */

    /* Wake up when the cache is expecting another value */
    *gpuWakeupMs = (unsigned int)((m_cacheIntervalUsec - (*now - gpu->lastCacheUsec)) / 1000);

    //printf("gpuWakeupMs %u, m_cacheIntervalUsec %lld, now %lld, m_lastCacheUsec %lld\n",
    //       *gpuWakeupMs, (long long)m_cacheIntervalUsec, (long long)*now, (long long)m_lastCacheUsec);

    if(*now - gpu->lastCacheUsec < m_cacheIntervalUsec)
        return;
    
    for(unsigned int fieldIdIndex = 0; fieldIdIndex < gpu->enabledFieldIds.size(); fieldIdIndex++)
    {
        switch(gpu->enabledFieldIds[fieldIdIndex])
        {
            case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_GR_ENGINE_ACTIVE, dcgmProfMetricGrActive, now, fvBuffer);
                break;

            case DCGM_FI_PROF_SM_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_SM_ACTIVE, dcgmProfMetricSmActive, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_SM_OCLWPANCY:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_SM_OCLWPANCY, dcgmProfMetricSmOclwpancy, now, fvBuffer);
                break;

            case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_PIPE_TENSOR_ACTIVE, dcgmProfMetricPipeTensorActive, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_DRAM_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_DRAM_ACTIVE, dcgmProfMetricDramActive, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_PIPE_FP64_ACTIVE, dcgmProfMetricPipeFp64Active, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_PIPE_FP32_ACTIVE, dcgmProfMetricPipeFp32Active, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
                CacheAvgDoubleValue(gpu, DCGM_FI_PROF_PIPE_FP16_ACTIVE, dcgmProfMetricPipeFp16Active, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_PCIE_TX_BYTES:
                CacheAvgInt64BandwidthValue(gpu, DCGM_FI_PROF_PCIE_TX_BYTES, dcgmProfMetricPcieTxBytes, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_PCIE_RX_BYTES:
                CacheAvgInt64BandwidthValue(gpu, DCGM_FI_PROF_PCIE_RX_BYTES, dcgmProfMetricPcieRxBytes, now, fvBuffer);
                break;

            case DCGM_FI_PROF_LWLINK_TX_BYTES:
                CacheAvgInt64BandwidthValue(gpu, DCGM_FI_PROF_LWLINK_TX_BYTES, dcgmProfMetricLwLinkTxBytes, now, fvBuffer);
                break;
            
            case DCGM_FI_PROF_LWLINK_RX_BYTES:
                CacheAvgInt64BandwidthValue(gpu, DCGM_FI_PROF_LWLINK_RX_BYTES, dcgmProfMetricLwLinkRxBytes, now, fvBuffer);
                break;
            
            default:
                PRINT_ERROR("%u", "Unimplemented fieldId: %u", gpu->enabledFieldIds[fieldIdIndex]);
                break;
        }
    }

    /* Clear all counters for this gpu */
    for(unsigned int i = 0; i < dcgmProfMetricCount; i++)
    {
        gpu->lopMetrics[i].values.clear();
    }
    
    gpu->lastCacheUsec = *now;
}

/*****************************************************************************/
unsigned int DcgmModuleProfiling::ReadAndCacheMetrics()
{
    timelib64_t now = timelib_usecSince1970();
    unsigned int minWakeupMs = 10000; /* Default to 10 seconds */
    unsigned int gpuWakeupMs;
    unsigned int gpuIndex;

    DcgmFvBuffer *fvBuffer = m_taskRunnerFvBuffer;
    fvBuffer->Clear();

    if(m_profilingIsPaused)
        PRINT_DEBUG("", "ReadAndCacheMetrics() called with profiling paused. All metric reading will be skipped.");

    for(gpuIndex = 0; gpuIndex < m_gpus.size(); gpuIndex++)
    {
        if(!m_profilingIsPaused)
        {
            gpuWakeupMs = DCGM_INT32_BLANK;
            ReadGpuMetrics(&m_gpus[gpuIndex], &gpuWakeupMs, &now);
            if(!DCGM_INT32_IS_BLANK(gpuWakeupMs))
            {
                minWakeupMs = DCGM_MIN(minWakeupMs, gpuWakeupMs);
            }
        }

        gpuWakeupMs = DCGM_INT32_BLANK;
        SendGpuMetricsToCache(&m_gpus[gpuIndex], &gpuWakeupMs, &now, fvBuffer);
        if(!DCGM_INT32_IS_BLANK(gpuWakeupMs))
        {
            minWakeupMs = DCGM_MIN(minWakeupMs, gpuWakeupMs);
        }
    }

    /* Send any FVs we buffered to the cache manager */

    size_t bufferSize = 0;
    size_t elementCount = 0;
    fvBuffer->GetSize(&bufferSize, &elementCount);
    if(elementCount > 0)
        mpCacheManager->AppendSamples(fvBuffer);
    fvBuffer->Clear();

    if(!minWakeupMs)
        minWakeupMs = 1; /* Sleep at least a ms. Returning 0 will make the task runner hibernate */

    return minWakeupMs;
}

/*****************************************************************************/
unsigned int DcgmModuleProfiling::RunOnce()
{
    return ReadAndCacheMetrics();
}

/*****************************************************************************/
class DcgmModuleProfilingTask : public DcgmTask
{
public:
    DcgmModuleProfilingTask(DcgmModuleProfiling *dcgmModuleProfiling, 
                            dcgm_module_command_header_t *moduleCommand) :
        m_dcgmReturn(DCGM_ST_OK), 
        m_moduleCommand(moduleCommand), 
        m_dcgmModuleProfiling(dcgmModuleProfiling)
    {
    }

    /*************************************************************************/
    ~DcgmModuleProfilingTask()
    {
    }

    /*************************************************************************/
    void Process()
    {
        m_dcgmReturn = m_dcgmModuleProfiling->ProcessMessageFromTaskRunner(m_moduleCommand);
        MarkDoneAndNotify();
    }

    /*************************************************************************/
    dcgmReturn_t GetDcgmReturn()
    {
        return m_dcgmReturn;
    }

    /*************************************************************************/
private:
    dcgmReturn_t m_dcgmReturn;                     /* Error code returned from Process() */
    dcgm_module_command_header_t *m_moduleCommand; /* Pointer to the module command owned by the caller */
    DcgmModuleProfiling *m_dcgmModuleProfiling;    /* Instance of DcgmModuleProfiling to call ProcessMessageReally on */ 
    /**********************************************************************/
};

/*****************************************************************************/
void DcgmModuleProfiling::UnwatchAll(void)
{
    for(unsigned int i = 0; i < m_gpus.size(); i++)
    {
        dcgm_module_prof_gpu_t *gpu = &m_gpus[i];

        PRINT_DEBUG("%u", "Clearing all perfworks watches for GPU %u", gpu->gpuId);

        /* Clear any previous watches for this GPU */
        ClearLopWatchesForGpu(gpu);

        gpu->enabledFieldIds.clear();
    }

    /* Clear ownership of the watches since there are none */
    m_watcherConnId = DCGM_CONNECTION_ID_NONE;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessPauseResume(dcgm_profiling_msg_pause_resume_t *msg)
{
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retVal = DCGM_ST_OK;

    dcgmReturn = CheckVersion(&msg->header, dcgm_profiling_msg_pause_resume_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    bool oldPauseState = m_profilingIsPaused;

    PRINT_INFO("%d %d", "Got request to change profiling paused state from %d -> %d", 
               oldPauseState, msg->pause);
    
    if(oldPauseState == msg->pause)
    {
        PRINT_DEBUG("", "Nothing to do.");
        return DCGM_ST_OK;
    }

    /* Set the new state before we possibly return early */
    m_profilingIsPaused = msg->pause;

    for(unsigned int i = 0; i < m_gpus.size(); i++)
    {
        dcgm_module_prof_gpu_t *gpu = &m_gpus[i];

        if(!gpu->lopGpu)
        {
            PRINT_DEBUG("%u", "gpu %u's lopGpu was NULL", gpu->gpuId);
            continue;
        }

        if(m_profilingIsPaused)
        {
            /* enabled -> disabled. DisableMetrics() handles being called unconditionally */
            dcgmReturn = gpu->lopGpu->DisableMetrics();
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u", "lopGpu->DisableMetrics() returned %d for gpuId %u", 
                            dcgmReturn, gpu->gpuId);
                retVal = dcgmReturn;
                /* Keep going. We have multiple GPUs to disable */
            }
            else
                PRINT_DEBUG("%u", "Disabled metrics for gpuId %u", gpu->gpuId);
        }
        else
        {
            /* disabled -> enabled. If we have fields enabled, then enable metrics */
            if(gpu->enabledFieldIds.size() < 1)
            {
                PRINT_DEBUG("%u", "Skipping EnableMetrics() for gpuId %u since no "
                            "metrics are lwrrently watched.",  gpu->gpuId);
                continue;
            }

            dcgmReturn = gpu->lopGpu->EnableMetrics();
            if(dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%d %u", "lopGpu->EnableMetrics() returned %d for gpuId %u", 
                            dcgmReturn, gpu->gpuId);
                retVal = dcgmReturn;
                /* Keep going. We have multiple GPUs to enable */
            }
            else
                PRINT_DEBUG("%u", "Enabled metrics for gpuId %u", gpu->gpuId);
        }
    }

    return retVal;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg)
{
    if(m_watcherConnId == DCGM_CONNECTION_ID_NONE || msg->connectionId != m_watcherConnId)
    {
        PRINT_DEBUG("%u %u", "Nothing to do for disconnected clientId %u, m_watcherConnId %u",
                    msg->connectionId, m_watcherConnId);
        return DCGM_ST_OK;
    }

    PRINT_INFO("%u", "Unwatching all perfworks metrics for disconnecting connId %u", msg->connectionId);
    UnwatchAll();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_CLIENT_DISCONNECT:
            retSt = ProcessClientDisconnect((dcgm_core_msg_client_disconnect_t *)moduleCommand);
            break;

        default:
            PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
            return DCGM_ST_BADPARAM;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    bool processInTaskRunner = false;

    if(moduleCommand->moduleId == DcgmModuleIdCore)
    {
        processInTaskRunner = true;
    }
    else if(moduleCommand->moduleId != DcgmModuleIdProfiling)
    {
        PRINT_ERROR("%u", "Unexpected module command for module %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }
    else /* Profiling module request */
    {
        switch(moduleCommand->subCommand)
        {
            /* If you want to add any module command to be processed inline instead of 
            in the task runner, add it as a separate case here */

            /* Messages to process on the task runner */
            case DCGM_PROFILING_SR_GET_MGS:
            case DCGM_PROFILING_SR_WATCH_FIELDS:
            case DCGM_PROFILING_SR_UNWATCH_FIELDS:
            case DCGM_PROFILING_SR_PAUSE_RESUME:
                {
                    processInTaskRunner = true;
                }
                break;

            default:
                PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
                return DCGM_ST_BADPARAM;
                break;
        }
    }

    if(processInTaskRunner)
    {
        /* Using a local since the task object is owned by the caller. If we
            ever have a multi-step task or one that is freed by the taskrunner, then
            we will need a new/delete pair for the task and will need to modify 
            DcgmModuleProfilingTask to make a copy of moduleCommand */
        DcgmModuleProfilingTask task(this, moduleCommand);
        dcgmReturn_t dcgmReturn = QueueTask(&task);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "dcgmReturn %d from QueueTask()", dcgmReturn);
            return dcgmReturn;
        }

        task.Wait();
        retSt = task.GetDcgmReturn();
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleProfiling::ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    if(moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else if(moduleCommand->moduleId != DcgmModuleIdProfiling)
    {
        PRINT_ERROR("%u", "Unexpected module command for module %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }
    else /* Profiling module request */
    {
        switch(moduleCommand->subCommand)
        {
            case DCGM_PROFILING_SR_GET_MGS:
                retSt = ProcessGetMetricGroups((dcgm_profiling_msg_get_mgs_t *)moduleCommand);
                break;

            case DCGM_PROFILING_SR_WATCH_FIELDS:
                retSt = ProcessWatchFields((dcgm_profiling_msg_watch_fields_t *)moduleCommand);
                break;
            
            case DCGM_PROFILING_SR_UNWATCH_FIELDS:
                retSt = ProcessUnwatchFields((dcgm_profiling_msg_unwatch_fields_t *)moduleCommand);
                break;
            
            case DCGM_PROFILING_SR_PAUSE_RESUME:
                retSt = ProcessPauseResume((dcgm_profiling_msg_pause_resume_t *)moduleCommand);
                break;

            default:
                PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
                return DCGM_ST_BADPARAM;
                break;
        }
    }

    return retSt;
}

/*****************************************************************************/
extern "C" DcgmModule *dcgm_alloc_module_instance(void)
{
    return (DcgmModule *)new DcgmModuleProfiling();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
