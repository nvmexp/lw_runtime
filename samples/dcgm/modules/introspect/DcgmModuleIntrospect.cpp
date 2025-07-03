#include "DcgmModuleIntrospect.h"
#include "dcgm_introspect_structs.h"
#include "dcgm_structs.h"
#include "logging.h"
#include "DcgmLogging.h"
#include "LwcmHostEngineHandler.h"

/*****************************************************************************/
DcgmModuleIntrospect::DcgmModuleIntrospect()
{
    mpMetadataManager = NULL;
    mpCacheManager = LwcmHostEngineHandler::Instance()->GetCacheManager();
    if(!mpCacheManager)
    {
        const char *errorStr = "DcgmModuleIntrospect was unable to find the cache manager.";
        PRINT_ERROR("%s", "%s", errorStr);
        throw runtime_error(errorStr);
    }
}

/*****************************************************************************/
DcgmModuleIntrospect::~DcgmModuleIntrospect()
{
    delete mpMetadataManager;
    mpCacheManager = 0; /* Not owned by us */
}

/*****************************************************************************/
static
DcgmMetadataManager::StatContext introspectLevelToStatContext(dcgmIntrospectLevel_t lvl)
{
    switch (lvl)
    {
        case DCGM_INTROSPECT_LVL_FIELD:
            return DcgmMetadataManager::STAT_CONTEXT_FIELD;
        case DCGM_INTROSPECT_LVL_FIELD_GROUP:
            return DcgmMetadataManager::STAT_CONTEXT_FIELD_GROUP;
        case DCGM_INTROSPECT_LVL_ALL_FIELDS:
            return DcgmMetadataManager::STAT_CONTEXT_ALL_FIELDS;
        default:
            return DcgmMetadataManager::STAT_CONTEXT_ILWALID;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::GetMemUsageForFields(dcgmIntrospectContext_t *context,
                                                        dcgmIntrospectFullMemory_t *memInfo,
                                                        int waitIfNoData)
{
    dcgmReturn_t st;

    if (context == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }
    if (memInfo == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    DcgmMetadataManager::StatContext statContext = introspectLevelToStatContext(context->introspectLvl);
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_ILWALID)
    {
        PRINT_ERROR("%d", "introspect level %d cannot be translated to a Metadata stat context",
                    context->introspectLvl);
        return DCGM_ST_BADPARAM;
    }

    int fieldScope = -1;
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(context->fieldId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%u", "%u is an invalid field", context->fieldId);
            return DCGM_ST_BADPARAM;
        }
        fieldScope = fieldMeta->scope;
    }

    // get aggregate info
    DcgmMetadataManager::ContextKey aggrContext(statContext, context->contextId, true);
    st = mpMetadataManager->GetBytesUsed(aggrContext, &memInfo->aggregateInfo.bytesUsed, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    // get global info
    memInfo->hasGlobalInfo = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD
        || (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD && fieldScope == DCGM_FS_GLOBAL))
    {
        DcgmMetadataManager::ContextKey globalContext(statContext, context->contextId, false, DCGM_FS_GLOBAL);
        st = mpMetadataManager->GetBytesUsed(globalContext, &memInfo->globalInfo.bytesUsed, waitIfNoData);

        // not watched isn't important since we already retrieved the aggregate info and something was watched
        if (DCGM_ST_OK != st && DCGM_ST_NOT_WATCHED != st)
            return st;

        memInfo->hasGlobalInfo = 1;
    }

    // get device info
    memInfo->gpuInfoCount = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD
        || (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD && fieldScope == DCGM_FS_DEVICE))
    {
        std::vector<unsigned int> gpuIds;
        mpCacheManager->GetGpuIds(1, gpuIds);

        // every time GPU info is found, insert it to the first open return slot
        size_t retIndex = 0;
        for (size_t i = 0; i < gpuIds.size(); ++i)
        {
            unsigned int gpuId = gpuIds.at(i);
            DcgmMetadataManager::ContextKey gpuContext(statContext, context->contextId, false, DCGM_FS_DEVICE, gpuId);
            st = mpMetadataManager->GetBytesUsed(gpuContext, &memInfo->gpuInfo[retIndex].bytesUsed, waitIfNoData);

            // not watched isn't important since we already retrieved the aggregate info and something was watched
            if (DCGM_ST_NO_DATA == st || DCGM_ST_NOT_WATCHED == st)
                continue;
            if (DCGM_ST_OK != st)
                return st;

            memInfo->gpuInfoCount++;
            memInfo->gpuIdsForGpuInfo[retIndex] = gpuId;
            retIndex++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::GetExecTimeForFields(dcgmIntrospectContext_t *context,
                                                        dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                        int waitIfNoData)
{
    dcgmReturn_t st;

    if (context == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }
    if (execTime == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    DcgmMetadataManager::StatContext statContext = introspectLevelToStatContext(context->introspectLvl);
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_ILWALID)
    {
        PRINT_ERROR("%d", "introspect level %d cannot be translated to a Metadata stat context",
                    context->introspectLvl);
        return DCGM_ST_BADPARAM;
    }

    int fieldScope = -1;
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(context->contextId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%llu", "%llu is an invalid field", context->contextId);
            return DCGM_ST_BADPARAM;
        }
        fieldScope = fieldMeta->scope;
    }

    // get aggregate info
    DcgmMetadataManager::ContextKey aggrContext(statContext, context->contextId, true);
    DcgmMetadataManager::ExecTimeInfo aggrExecTime;

    st = mpMetadataManager->GetExecTime(aggrContext, &aggrExecTime, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    CopyFieldsExecTime(execTime->aggregateInfo, aggrExecTime);

    // get global info
    execTime->hasGlobalInfo = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD
        || (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD && fieldScope == DCGM_FS_GLOBAL))
    {
        DcgmMetadataManager::ContextKey globalContext(statContext, context->contextId, false, DCGM_FS_GLOBAL);
        DcgmMetadataManager::ExecTimeInfo globalExecTime;

        st = mpMetadataManager->GetExecTime(globalContext, &globalExecTime, waitIfNoData);

        // not watched isn't important since we already retrieved the aggregate info and something was watched
        if (DCGM_ST_OK != st && DCGM_ST_NOT_WATCHED != st)
            return st;

        CopyFieldsExecTime(execTime->globalInfo, globalExecTime);
        execTime->hasGlobalInfo = 1;
    }

    // get device info
    execTime->gpuInfoCount = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD
        || (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD && fieldScope == DCGM_FS_DEVICE))
    {
        std::vector<unsigned int> gpuIds;
        this->mpCacheManager->GetGpuIds(1, gpuIds);

        unsigned int retIndex = 0;
        for (size_t i = 0; i < gpuIds.size(); ++i)
        {
            unsigned int gpuId = gpuIds.at(i);
            DcgmMetadataManager::ContextKey gpuContext(statContext, context->contextId, false, DCGM_FS_DEVICE, gpuId);
            DcgmMetadataManager::ExecTimeInfo gpuExecTime;

            st = mpMetadataManager->GetExecTime(gpuContext, &gpuExecTime, waitIfNoData);

            // not watched isn't important since we already retrieved the aggregate info and something was watched
            if (DCGM_ST_NO_DATA == st || DCGM_ST_NOT_WATCHED == st)
                continue;
            if (DCGM_ST_OK != st)
                return st;

            // every time GPU info is found, insert it to the first open return slot
            execTime->gpuInfoCount++;
            execTime->gpuIdsForGpuInfo[retIndex] = gpuId;
            CopyFieldsExecTime(execTime->gpuInfo[retIndex], gpuExecTime);
            retIndex++;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleIntrospect::GetMemUsageForHostengine(dcgmIntrospectMemory_t *memInfo, int waitIfNoData)
{
    DcgmMetadataManager::ContextKey context(DcgmMetadataManager::STAT_CONTEXT_PROCESS);
    return mpMetadataManager->GetBytesUsed(context, &memInfo->bytesUsed, waitIfNoData);
}

dcgmReturn_t DcgmModuleIntrospect::GetCpuUtilizationForHostengine(dcgmIntrospectCpuUtil_t *cpuUtil, int waitIfNoData)
{
    dcgmReturn_t st;

    DcgmMetadataManager::CpuUtil mgrCpuUtil;
    st = mpMetadataManager->GetCpuUtilization(&mgrCpuUtil, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    cpuUtil->kernel = mgrCpuUtil.kernel;
    cpuUtil->user = mgrCpuUtil.user;
    cpuUtil->total = mgrCpuUtil.total;

    return DCGM_ST_OK;
}

void DcgmModuleIntrospect::CopyFieldsExecTime(dcgmIntrospectFieldsExecTime_t &execTime,
                                              const DcgmMetadataManager::ExecTimeInfo &metadataExecTime)
{
    execTime.meanUpdateFreqUsec = metadataExecTime.meanFrequencyUsec;
    execTime.recentUpdateUsec = metadataExecTime.recentUpdateUsec;
    execTime.totalEverUpdateUsec = metadataExecTime.totalEverUpdateUsec;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataFieldsExecTime(dcgm_introspect_msg_fields_exec_time_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_fields_exec_time_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->execTime.version != dcgmIntrospectFullFieldsExecTime_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectFullFieldsExecTime_version, msg->execTime.version);
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn = GetExecTimeForFields(&msg->context, &msg->execTime, msg->waitIfNoData);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataFieldsMemUsage(dcgm_introspect_msg_fields_mem_usage_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_fields_mem_usage_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->memoryInfo.version != dcgmIntrospectFullMemory_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectFullMemory_version, msg->memoryInfo.version);
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn = GetMemUsageForFields(&msg->context, &msg->memoryInfo, msg->waitIfNoData);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataHostEngineCpuUtil(dcgm_introspect_msg_he_cpu_util_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_cpu_util_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->cpuUtil.version != dcgmIntrospectCpuUtil_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectCpuUtil_version, msg->cpuUtil.version);
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn = GetCpuUtilizationForHostengine(&msg->cpuUtil, msg->waitIfNoData);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataHostEngineMemUsage(dcgm_introspect_msg_he_mem_usage_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_mem_usage_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->memoryInfo.version != dcgmIntrospectMemory_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectMemory_version, msg->memoryInfo.version);
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn = GetMemUsageForHostengine(&msg->memoryInfo, msg->waitIfNoData);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataStateSetRunInterval(dcgm_introspect_msg_set_interval_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_set_interval_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    dcgmReturn = mpMetadataManager->SetRunInterval(msg->runIntervalMs);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataStateToggle(dcgm_introspect_msg_toggle_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_toggle_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    switch (msg->enabledStatus)
    {
        case (DCGM_INTROSPECT_STATE_ENABLED):
            if (NULL == mpMetadataManager)
            {
                mpMetadataManager = new DcgmMetadataManager(mpCacheManager);
                int ret = mpMetadataManager->Start();
                if(ret)
                {
                    std::stringstream ss;
                    ss << "IntrospectionManager Start Failed. Error: " << ret;
                    throw std::runtime_error(ss.str());
                }
                PRINT_DEBUG("", "IntrospectionManager started");
            }
            else
            {
                PRINT_DEBUG("", "IntrospectionManager already started");
            }
            dcgmReturn = DCGM_ST_OK;
            break;
        case (DCGM_INTROSPECT_STATE_DISABLED):
            if (NULL == mpMetadataManager)
            {
                PRINT_DEBUG("", "IntrospectionManager already disabled");
            }
            else
            {
                delete mpMetadataManager;
                mpMetadataManager = NULL;
                PRINT_DEBUG("", "IntrospectionManager disabled");
            }
            dcgmReturn = DCGM_ST_OK;
            break;
        default:
            PRINT_ERROR("%d", "%d is an unkown state to set metadata collection to", msg->enabledStatus);
            dcgmReturn = DCGM_ST_BADPARAM;
            break;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataUpdateAll(dcgm_introspect_msg_update_all_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;
    
    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_update_all_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */
    
    dcgmReturn = mpMetadataManager->UpdateAll(msg->waitForUpdate);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::VerifyMetadataEnabled()
{
    if (!mpMetadataManager)
    {
        PRINT_ERROR("", "Trying to access metadata APIs but metadata gathering is not enabled");
        return DCGM_ST_NOT_CONFIGURED;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch(moduleCommand->subCommand)
    {
        case DCGM_INTROSPECT_SR_STATE_TOGGLE:
            retSt = ProcessMetadataStateToggle((dcgm_introspect_msg_toggle_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL:
            retSt = ProcessMetadataStateSetRunInterval((dcgm_introspect_msg_set_interval_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_UPDATE_ALL:
            retSt = ProcessMetadataUpdateAll((dcgm_introspect_msg_update_all_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE:
            retSt = ProcessMetadataHostEngineMemUsage((dcgm_introspect_msg_he_mem_usage_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL:
            retSt = ProcessMetadataHostEngineCpuUtil((dcgm_introspect_msg_he_cpu_util_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE:
            retSt = ProcessMetadataFieldsMemUsage((dcgm_introspect_msg_fields_mem_usage_t *)moduleCommand);
            break;

        case DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME:
            retSt = ProcessMetadataFieldsExecTime((dcgm_introspect_msg_fields_exec_time_t *)moduleCommand);
            break;

        default:
            PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
            return DCGM_ST_BADPARAM;
            break;
    }

    return retSt;
}

/*****************************************************************************/
extern "C" DcgmModule *dcgm_alloc_module_instance(void)
{
    return (DcgmModule *)new DcgmModuleIntrospect();
}

/*****************************************************************************/
extern "C" void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete(freeMe);
}

/*****************************************************************************/
