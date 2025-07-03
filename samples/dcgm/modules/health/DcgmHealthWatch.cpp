#include "DcgmHealthWatch.h"
#include "dcgm_agent_internal.h"
#include "timelib.h"
#include <sstream>
#include <stdexcept>
#include "logging.h"
#include "dcgm_errors.h"

// Adds a watch for the specified field that will poll every 10 seconds for the last hour's events
#define ADD_WATCH(fieldId)                                                        \
    do{                                                                           \
        ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, fieldId, 10000000, 3600.0, 0, watcher, false); \
        if (DCGM_ST_OK != ret)                                                    \
        {                                                                         \
            std::stringstream ss;                                                 \
            ss << "Failed to set watch for field " << fieldId << " on GPU " << gpuId; \
            PRINT_ERROR("%s", "%s", ss.str().c_str());                            \
            return ret;                                                           \
        }                                                                         \
    } while (0)

/*****************************************************************************/
DcgmHealthWatch::DcgmHealthWatch(LwcmGroupManager *gm, DcgmCacheManager *cm)
{
    mpGroupManager = gm;
    mpCacheManager = cm;

    m_mutex = new DcgmMutex(0);

    mGroupWatchState.clear();

    // don't worry about an error here.  But need to check m_etblLwcmEngineInternal
    // in each function that uses it
    dcgmInternalGetExportTable((const void**)&m_etblLwcmEngineInternal,
                        &ETID_DCGMEngineInternal);


    BuildFieldLists();
}

/*****************************************************************************/
DcgmHealthWatch::~DcgmHealthWatch()
{
    if(m_mutex)
    {
        delete(m_mutex);
        m_mutex = 0;
    }
}

/*****************************************************************************/
void DcgmHealthWatch::BuildFieldLists(void)
{
    // all the non-fatal error field ids.
    m_lwSwitchNonFatalFieldIds.push_back(DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS);

    // all the fatal error field ids.
    m_lwSwitchFatalFieldIds.push_back(DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS);
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetLwSwitchWatches(std::vector<unsigned int> &groupSwitchIds,
                                                 dcgmHealthSystems_t systems,
                                                 DcgmWatcher watcher)
{
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int>::iterator switchIter;
    unsigned int i;
    long long watchFreq = 10000000; /* Update every 10 seconds */
    double maxKeepAge = 3600.0;     /* Keep samples for an hour */
    
    for(switchIter = groupSwitchIds.begin(); switchIter != groupSwitchIds.end(); ++switchIter)
    {
        if(systems & DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL)
        {
            for(i = 0; i < m_lwSwitchNonFatalFieldIds.size(); i++)
            {
                dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_SWITCH, 
                                                *switchIter, m_lwSwitchNonFatalFieldIds[i], 
                                                watchFreq, maxKeepAge, 0, watcher, false);
                if(dcgmReturn != DCGM_ST_OK)
                {
                    PRINT_ERROR("%d", "Error %d from AddEntityFieldWatch() for LwSwitch fields", 
                                (int)dcgmReturn);
                    return dcgmReturn;
                }
            }
        }

        if(systems & DCGM_HEALTH_WATCH_LWSWITCH_FATAL)
        {
            for(i = 0; i < m_lwSwitchFatalFieldIds.size(); i++)
            {
                dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_SWITCH, 
                                                *switchIter, m_lwSwitchFatalFieldIds[i],
                                                watchFreq, maxKeepAge, 0, watcher, false);
                if(dcgmReturn != DCGM_ST_OK)
                {
                    PRINT_ERROR("%d", "Error %d from AddEntityFieldWatch() for LwSwitch fields", 
                                (int)dcgmReturn);
                    return dcgmReturn;
                }
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetWatches(unsigned int groupId, dcgmHealthSystems_t systems,
                                         dcgm_connection_id_t connectionId)
{
    unsigned int index, gpuId, gpuIdIndex;
    dcgmReturn_t ret = DCGM_ST_OK;
    std::vector<unsigned int>groupGpuIds;
    std::vector<unsigned int>groupSwitchIds;
    DcgmWatcher watcher(DcgmWatcherTypeHealthWatch, connectionId);
    std::vector<dcgmGroupEntityPair_t>entities;

    ret = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got st %d from GetGroupEntities()", ret);
        return ret;
    }

    dcgm_mutex_lock(m_mutex);
    mGroupWatchState[groupId] = systems;
    dcgm_mutex_unlock(m_mutex);

    /* Capture the entities that are GPUs as a separate list */
    for (index = 0; index < entities.size(); index++) 
    {
        if (entities[index].entityGroupId == DCGM_FE_GPU)
            groupGpuIds.push_back(entities[index].entityId);
        else if(entities[index].entityGroupId == DCGM_FE_SWITCH)
            groupSwitchIds.push_back(entities[index].entityId);
    }

    if(groupSwitchIds.size() > 0)
        ret = SetLwSwitchWatches(groupSwitchIds, systems, watcher);

    for (gpuIdIndex = 0; gpuIdIndex < groupGpuIds.size(); gpuIdIndex++)
    {
        gpuId = groupGpuIds[gpuIdIndex];
        if(gpuId >= DCGM_MAX_NUM_DEVICES)
        {
            PRINT_ERROR("%u", "gpuId %u out of range", gpuId);
            break;
        }

        for (index = 0; index < DCGM_HEALTH_WATCH_COUNT_V2; index++)
        {
            unsigned int bit = 1 << index;
            switch (bit)
            {
                case DCGM_HEALTH_WATCH_PCIE:
                    ret = SetPcie(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                case DCGM_HEALTH_WATCH_MEM:
                    ret = SetMem(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                case DCGM_HEALTH_WATCH_INFOROM:
                    ret = SetInforom(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                case DCGM_HEALTH_WATCH_THERMAL:
                    ret = SetThermal(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                case DCGM_HEALTH_WATCH_POWER:
                    ret = SetPower(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                case DCGM_HEALTH_WATCH_LWLINK:
                    ret = SetLWLink(gpuId, (systems&bit) ? true : false, watcher);
                    break;
                default: // ignore everything else for now
                    break;
            }
            if (DCGM_ST_OK != ret)
            {
                PRINT_ERROR("%d %u %u", "Error %d from bit %u, gpuId %u",
                            (int)ret, bit, groupGpuIds[gpuIdIndex]);
                break; // exit on error
            }
        }
    }
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::GetWatches(unsigned int groupId, dcgmHealthSystems_t *systems)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    groupWatchTable_t::iterator groupWatchIter;
    std::vector<dcgmGroupEntityPair_t>entities;

    ret = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got st %d from GetGroupEntities()", ret);
        return ret;
    }

    dcgm_mutex_lock(m_mutex);
    groupWatchIter = mGroupWatchState.find(groupId);
    if(groupWatchIter == mGroupWatchState.end())
    {
        *systems = (dcgmHealthSystems_enum)0;
    }
    else
    {
        *systems = groupWatchIter->second;
    }

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}
    
/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorWatchesForGpu(unsigned int gpuId, long long startTime, long long endTime,
                                                   dcgmHealthSystems_t healthSystemsMask,
                                                   dcgmHealthResponse_v1 *response)

{
    dcgmReturn_t ret = DCGM_ST_OK;
    
    for (unsigned int index = 0; index < DCGM_HEALTH_WATCH_COUNT_V1; index++)
    {
        unsigned int bit = 1 << index;
        dcgmReturn_t tmpRet = DCGM_ST_OK;

        switch (bit)
        {
            case DCGM_HEALTH_WATCH_PCIE:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorPcie(DCGM_FE_GPU, gpuId, startTime, endTime,  response);
                break;
            case DCGM_HEALTH_WATCH_MEM:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorMem(DCGM_FE_GPU, gpuId,  startTime, endTime,  response);
                break;
            case DCGM_HEALTH_WATCH_INFOROM:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorInforom(DCGM_FE_GPU, gpuId,  startTime, endTime,  response);
                break;
            case DCGM_HEALTH_WATCH_THERMAL:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorThermal(DCGM_FE_GPU, gpuId, startTime, endTime,   response);
                break;
            case DCGM_HEALTH_WATCH_POWER:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorPower(DCGM_FE_GPU, gpuId,  startTime, endTime,  response);
                break;
            case DCGM_HEALTH_WATCH_LWLINK:
                if (bit & healthSystemsMask)
                    tmpRet = MonitorLWLink(DCGM_FE_GPU, gpuId, startTime, endTime,  response);
                break;
            default: // ignore everything else for now, other bugs
                break;
        }

        if ((ret == DCGM_ST_OK) && (tmpRet != DCGM_ST_OK))
            ret = tmpRet;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorWatchesV1(unsigned int groupId, long long startTime, long long endTime, dcgmHealthResponse_v1 *response)
{
    unsigned int index;
    dcgmReturn_t ret = DCGM_ST_OK;
    unsigned int gpuIdIndex;
    std::vector<dcgmGroupEntityPair_t>entities;
    dcgmHealthSystems_t healthSystemsMask = (dcgmHealthSystems_t)0; /* Cached version of this group's watch mask */
    std::vector<unsigned int>gpuIds;

    /* Handle BLANK start-time and end-time */
    if(DCGM_INT64_IS_BLANK(startTime))
        startTime = 0;
    if(DCGM_INT64_IS_BLANK(endTime))
        endTime = 0;

    ret = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got st %d from GetGroupEntities()", ret);
        return ret;
    }

    dcgm_mutex_lock(m_mutex);
    groupWatchTable_t::iterator groupWatchIter = mGroupWatchState.find(groupId);
    if(groupWatchIter != mGroupWatchState.end())
    {
        healthSystemsMask = groupWatchIter->second;
        PRINT_DEBUG("%X %u", "Found health systems mask %X for groupId %u", 
                    (unsigned int)healthSystemsMask, groupId);
    }
    else
    {
        PRINT_DEBUG("%u", "Found NO health systems mask for groupId %u", 
                    groupId);
    }
    dcgm_mutex_unlock(m_mutex);

    if(healthSystemsMask == 0)
        return DCGM_ST_OK; /* This is the same as walking over the loops below and doing nothing */

    /* Capture the entities that are GPUs as a separate list */
    for (index = 0; index < entities.size(); index++) 
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[index].entityId);
    }

    response->version = dcgmHealthResponse_version1;
    for(gpuIdIndex = 0; gpuIdIndex < gpuIds.size(); gpuIdIndex++)
    {
        dcgmReturn_t tmpRet = MonitorWatchesForGpu(gpuIds[gpuIdIndex], startTime, endTime, healthSystemsMask, response);

        if ((ret == DCGM_ST_OK) && (tmpRet != DCGM_ST_OK))
            ret = tmpRet;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorWatchesV2(unsigned int groupId, long long startTime, long long endTime,
                                               void *response)
{
    unsigned int index, entityIndex;
    dcgmReturn_t ret = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t>entities;
    dcgmHealthSystems_t healthSystemsMask = (dcgmHealthSystems_t)0; /* Cached version of this group's watch mask */
    std::vector<unsigned int>gpuIds;

    /* Handle BLANK start-time and end-time */
    if(DCGM_INT64_IS_BLANK(startTime))
        startTime = 0;
    if(DCGM_INT64_IS_BLANK(endTime))
        endTime = 0;

    ret = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got st %d from GetGroupEntities()", ret);
        return ret;
    }

    dcgm_mutex_lock(m_mutex);
    groupWatchTable_t::iterator groupWatchIter = mGroupWatchState.find(groupId);
    if(groupWatchIter != mGroupWatchState.end())
    {
        healthSystemsMask = groupWatchIter->second;
        PRINT_DEBUG("%X %u", "Found health systems mask %X for groupId %u", 
                    (unsigned int)healthSystemsMask, groupId);
    }
    else
    {
        PRINT_DEBUG("%u", "Found NO health systems mask for groupId %u", 
                    groupId);
    }
    dcgm_mutex_unlock(m_mutex);

    if(healthSystemsMask == 0)
        return DCGM_ST_OK; /* This is the same as walking over the loops below and doing nothing */

    /* Capture the entities that are GPUs as a separate list */
    for (index = 0; index < entities.size(); index++) 
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[index].entityId);
    }

    for(entityIndex = 0; entityIndex < entities.size(); entityIndex++)
    {
        dcgm_field_entity_group_t entityGroupId = entities[entityIndex].entityGroupId;
        dcgm_field_eid_t entityId = entities[entityIndex].entityId;

        for (index = 0; index < DCGM_HEALTH_WATCH_COUNT_V2; index++)
        {
            unsigned int bit = 1 << index;

            switch (bit)
            {
                case DCGM_HEALTH_WATCH_PCIE:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorPcie(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_MEM:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorMem(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_INFOROM:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorInforom(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_THERMAL:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorThermal(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_POWER:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorPower(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_LWLINK:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_GPU)
                        ret = MonitorLWLink(entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_SWITCH)
                        ret = MonitorLwSwitchErrorCounts(false, entityGroupId, entityId, startTime, endTime, response);
                    break;
                case DCGM_HEALTH_WATCH_LWSWITCH_FATAL:
                    if (bit & healthSystemsMask && entityGroupId == DCGM_FE_SWITCH)
                        ret = MonitorLwSwitchErrorCounts(true, entityGroupId, entityId, startTime, endTime, response);
                    break;
                default:
                    // reduce the logging level as this may pollute the log file if unsupported fields are watched continuously.
                    PRINT_DEBUG("%u", "Unhandled health bit %u", bit);
                    break;
            }
        }
    }

    return ret;
}

/*****************************************************************************/
void DcgmHealthWatch::SetResponse(dcgm_field_entity_group_t entityGroupId, 
                                  dcgm_field_eid_t entityId, 
                                  dcgmHealthWatchResults_t status, 
                                  dcgmHealthSystems_t system, DcgmError &d, 
                                  void *response, bool newIncidentRecord)
{
    unsigned int version = *(unsigned int *)response;

    if (version == dcgmHealthResponse_version1)
    {
        SetResponseV1(entityGroupId, entityId, status, system, d.GetMessage().c_str(),
                      static_cast<dcgmHealthResponse_v1 *>(response), newIncidentRecord);
    }
    else if (version == dcgmHealthResponse_version2)
    {
        SetResponseV2(entityGroupId, entityId, status, system, d.GetMessage().c_str(),
                      static_cast<dcgmHealthResponse_v2 *>(response), newIncidentRecord);
    }
    else if (version == dcgmHealthResponse_version3)
    {
        SetResponseV3(entityGroupId, entityId, status, system, d,
                      reinterpret_cast<dcgmHealthResponse_v3 *>(response), newIncidentRecord);
    }
    else
    {
        PRINT_ERROR("%X", "Unhandled response version x%X", version);
    }
}

/*****************************************************************************/
void DcgmHealthWatch::RecordNewIncidentV1(dcgmHealthResponse_v1 *response, unsigned int gpuIndex,
                                          dcgmHealthWatchResults_t status, dcgmHealthSystems_t system,
                                          const char *description)
{
    unsigned int incidentCount = response->gpu[gpuIndex].incidentCount;
    response->gpu[gpuIndex].incidentCount++;
    response->gpu[gpuIndex].systems[incidentCount].system = system;
    response->gpu[gpuIndex].systems[incidentCount].health = status;
    snprintf(response->gpu[gpuIndex].systems[incidentCount].errorString,
             sizeof(response->gpu[gpuIndex].systems[incidentCount].errorString), "%s", description);
}

/*****************************************************************************/
void DcgmHealthWatch::SetResponseV1(dcgm_field_entity_group_t entityGroupId, 
                                    dcgm_field_eid_t entityId, dcgmHealthWatchResults_t status, 
                                    dcgmHealthSystems_t system, const char *description, 
                                    dcgmHealthResponse_v1 *response, bool newIncidentRecord)
{
        unsigned int gpuId = entityId;

        if(entityGroupId != DCGM_FE_GPU)
            return; /* V1 requests only handle GPUs */

        // no need to check if these values are here... already set by the helper on the API side
        if (response->overallHealth < status) // set the overall health to status if not already there or worse
            response->overallHealth = status;

        unsigned int gpuIndex; /* Index of the GPU to modify in response->gpu[] */

        bool found = false;

        for (unsigned int i = 0; i < response->gpuCount; i++) // check to see if there is already a record there
        {
            if (response->gpu[i].gpuId == gpuId)
            {
                found = true; 
                gpuIndex = i;
                break;
            }
        }
        
        if (!found) // add one if a record was not found
        {
            gpuIndex = response->gpuCount;
            response->gpuCount += 1;
        }

        // fill in appropriate fields
        response->gpu[gpuIndex].gpuId = gpuId; // potentially redundant
        if(response->gpu[gpuIndex].overallHealth < status)
            response->gpu[gpuIndex].overallHealth = status;

        // if newIncidentRecord is true then go ahead and increase the incident count
        // it is implied that if it is false, this function is being called within the
        // same series of checks so just add the warning to the string
        if (newIncidentRecord)
        {
            RecordNewIncidentV1(response, gpuIndex, status, system, description);
        }
        else
        {
            found = false;
            int systemIndex = 0;
            for (int systemIndex = 0; systemIndex < DCGM_HEALTH_WATCH_COUNT_V1; systemIndex++)
            {
                if(response->gpu[gpuIndex].systems[systemIndex].system == system)
                {
                    found = true;
                    break;
                }
            }

            if (false == found)
            {
                if (response->gpu[gpuIndex].incidentCount >= DCGM_HEALTH_WATCH_COUNT_V1)
                {
                    PRINT_ERROR("%u", "System %u not found and cannot be inserted as a health check issue.",
                                static_cast<unsigned int>(system));
                }
                else
                {
                    PRINT_DEBUG("%u", 
                                "System %u was reported as a second incident, but wasn't found. Inserting as new",
                                static_cast<unsigned int>(system));
                    RecordNewIncidentV1(response, gpuIndex, status, system, description);
                }
            }
            else
            {
                std::string lwrrentErrorString = response->gpu[gpuIndex].systems[systemIndex].errorString;
                std::stringstream ss;
                ss << lwrrentErrorString << std::endl << description;
                snprintf(response->gpu[gpuIndex].systems[systemIndex].errorString, 
                         sizeof(response->gpu[gpuIndex].systems[systemIndex].errorString),
                         "%s", lwrrentErrorString.c_str());
            }

            if (response->gpu[gpuIndex].systems[systemIndex].health < status)
                response->gpu[gpuIndex].systems[systemIndex].health = status;
        }

        if (status == DCGM_HEALTH_RESULT_WARN)
            PRINT_WARNING("%s", "%s", description);
        else if (status == DCGM_HEALTH_RESULT_FAIL)
            PRINT_ERROR("%s", "%s", description);

}
    
/*****************************************************************************/
void DcgmHealthWatch::RecordNewIncidentV2(dcgmHealthResponse_v2 *response, unsigned int entityIndex,
                                          dcgmHealthWatchResults_t status, dcgmHealthSystems_t system,
                                          const char *description)
{
    unsigned int incidentCount = response->entities[entityIndex].incidentCount;
    response->entities[entityIndex].incidentCount++;
    response->entities[entityIndex].systems[incidentCount].system = system;
    response->entities[entityIndex].systems[incidentCount].health = status;
    snprintf(response->entities[entityIndex].systems[incidentCount].errorString,
             sizeof(response->entities[entityIndex].systems[incidentCount].errorString),
             "%s", description);
}

void DcgmHealthWatch::RecordErrorV3(dcgmHealthResponse_v3 *response, unsigned int entityIndex,
                                    unsigned int incidentIndex, const DcgmError &d)
{
    unsigned int  errorIndex = response->entities[entityIndex].systems[incidentIndex].errorCount;
    const char   *errMsg = d.GetMessage().c_str();
    unsigned int  errCode = d.GetCode();

    if (errorIndex > 3)
    {
        // Cannot record more than 4 entries
        errMsg = DCGM_FR_TOO_MANY_ERRORS_MSG;
        errCode = DCGM_FR_TOO_MANY_ERRORS;
        errorIndex = 3;
    }

    response->entities[entityIndex].systems[incidentIndex].errors[errorIndex].code = errCode;
    snprintf(response->entities[entityIndex].systems[incidentIndex].errors[errorIndex].msg,
             sizeof(response->entities[entityIndex].systems[incidentIndex].errors[errorIndex].msg),
             "%s", errMsg);

    response->entities[entityIndex].systems[incidentIndex].errorCount++;
}


void DcgmHealthWatch::RecordNewIncidentV3(dcgmHealthResponse_v3 *response, unsigned int entityIndex,
                                          dcgmHealthWatchResults_t status, dcgmHealthSystems_t system,
                                          const DcgmError &d)
{
    unsigned int incidentCount = response->entities[entityIndex].incidentCount;
    response->entities[entityIndex].incidentCount++;
    response->entities[entityIndex].systems[incidentCount].system = system;
    response->entities[entityIndex].systems[incidentCount].health = status;

    RecordErrorV3(response, entityIndex, incidentCount, d);
}

/*****************************************************************************/
void DcgmHealthWatch::SetResponseV2(dcgm_field_entity_group_t entityGroupId, 
                                    dcgm_field_eid_t entityId, 
                                    dcgmHealthWatchResults_t status, 
                                    dcgmHealthSystems_t system, const char *description, 
                                    dcgmHealthResponse_v2 *response, bool newIncidentRecord)
{
        // no need to check if these values are here... already set by the helper on the API side
        if (response->overallHealth < status) // set the overall health to status if not already there or worse
            response->overallHealth = status;

        unsigned int entityIndex; /* Index of the entity to modify in response->entities[] */

        bool found = false;

        // check to see if there is already a record for this entity
        for (unsigned int i = 0; i < response->entityCount; i++) 
        {
            if (response->entities[i].entityGroupId == entityGroupId &&
                response->entities[i].entityId == entityId)
            {
                found = true; 
                entityIndex = i;
                break;
            }
        }
        
        if (!found) // add one if a record was not found
        {
            /* Prevent buffer overflow */
            if(response->entityCount >= DCGM_GROUP_MAX_ENTITIES)
            {
                PRINT_ERROR("%d", "SetResponseV2 ran out of space for new entities at %d.",
                            DCGM_GROUP_MAX_ENTITIES);
                return;
            }

            entityIndex = response->entityCount;
            response->entityCount += 1;
        }

        // fill in appropriate fields
        response->entities[entityIndex].entityGroupId = entityGroupId;
        response->entities[entityIndex].entityId = entityId;
        if(response->entities[entityIndex].overallHealth < status)
            response->entities[entityIndex].overallHealth = status;

        // if newIncidentRecord is true then go ahead and increase the incident count
        // it is implied that if it is false, this function is being called within the
        // same series of checks so just add the warning to the string
        if (newIncidentRecord)
        {
            RecordNewIncidentV2(response, entityIndex, status, system, description);
        }
        else
        {
            found = false;
            int systemIndex = 0;
            for (int systemIndex = 0; systemIndex < DCGM_HEALTH_WATCH_COUNT_V2; systemIndex++)
            {
                if(response->entities[entityIndex].systems[systemIndex].system == system)
                {
                    found = true;
                    break;
                }
            }

            if (false == found)
            {
                if (response->entities[entityIndex].incidentCount >= DCGM_HEALTH_WATCH_COUNT_V2)
                {
                    PRINT_ERROR("%u", "System %u not found and cannot be inserted as a health check issue.",
                                static_cast<unsigned int>(system));
                }
                else
                {
                    PRINT_DEBUG("%u", 
                                "System %u was reported as a second incident, but wasn't found. Inserting as new",
                                static_cast<unsigned int>(system));
                    RecordNewIncidentV2(response, entityIndex, status, system, description);
                }
            }
            else
            {
                std::string lwrrentErrorString = response->entities[entityIndex].systems[systemIndex].errorString;
                std::stringstream ss;
                ss << lwrrentErrorString << std::endl << description;
                snprintf(response->entities[entityIndex].systems[systemIndex].errorString,
                         sizeof(response->entities[entityIndex].systems[systemIndex].errorString), 
                         "%s", lwrrentErrorString.c_str());
            }

            if (response->entities[entityIndex].systems[systemIndex].health < status)
                response->entities[entityIndex].systems[systemIndex].health = status;

        }
        if (status == DCGM_HEALTH_RESULT_WARN)
            PRINT_WARNING("%s", "%s", description);
        else if (status == DCGM_HEALTH_RESULT_FAIL)
            PRINT_ERROR("%s", "%s", description);

}

/*****************************************************************************/
void DcgmHealthWatch::SetResponseV3(dcgm_field_entity_group_t entityGroupId, 
                                    dcgm_field_eid_t entityId, 
                                    dcgmHealthWatchResults_t status, 
                                    dcgmHealthSystems_t system, const DcgmError &d, 
                                    dcgmHealthResponse_v3 *response, bool newIncidentRecord)
{
    // no need to check if these values are here... already set by the helper on the API side
    if (response->overallHealth < status) // set the overall health to status if not already there or worse
    {
        response->overallHealth = status;
    }

    unsigned int entityIndex; /* Index of the entity to modify in response->entities[] */

    bool found = false;

    // check to see if there is already a record for this entity
    for (unsigned int i = 0; i < response->entityCount; i++) 
    {
        if (response->entities[i].entityGroupId == entityGroupId &&
            response->entities[i].entityId == entityId)
        {
            found = true; 
            entityIndex = i;
            break;
        }
    }
    
    if (!found) // add one if a record was not found
    {
        /* Prevent buffer overflow */
        if (response->entityCount >= DCGM_GROUP_MAX_ENTITIES)
        {
            PRINT_ERROR("%d", "SetResponseV2 ran out of space for new entities at %d.",
                        DCGM_GROUP_MAX_ENTITIES);
            return;
        }

        entityIndex = response->entityCount;
        response->entityCount += 1;
    }

    // fill in appropriate fields
    response->entities[entityIndex].entityGroupId = entityGroupId;
    response->entities[entityIndex].entityId = entityId;
    if (response->entities[entityIndex].overallHealth < status)
    {
        response->entities[entityIndex].overallHealth = status;
    }

    // if newIncidentRecord is true then go ahead and increase the incident count
    // it is implied that if it is false, this function is being called within the
    // same series of checks so just add the warning to the string
    if (newIncidentRecord)
    {
        RecordNewIncidentV3(response, entityIndex, status, system, d);
    }
    else
    {
        found = false;
        int systemIndex = 0;
        for (int systemIndex = 0; systemIndex < DCGM_HEALTH_WATCH_COUNT_V2; systemIndex++)
        {
            if(response->entities[entityIndex].systems[systemIndex].system == system)
            {
                found = true;
                break;
            }
        }

        if (false == found)
        {
            if (response->entities[entityIndex].incidentCount >= DCGM_HEALTH_WATCH_COUNT_V2)
            {
                PRINT_ERROR("%u", "System %u not found and cannot be inserted as a health check issue.",
                            static_cast<unsigned int>(system));
            }
            else
            {
                PRINT_DEBUG("%u", 
                            "System %u was reported as a second incident, but wasn't found. Inserting as new",
                            static_cast<unsigned int>(system));
                RecordNewIncidentV3(response, entityIndex, status, system, d);
            }
        }
        else
        {
            RecordErrorV3(response, entityIndex, systemIndex, d);
        }

        if (response->entities[entityIndex].systems[systemIndex].health < status)
        {
            response->entities[entityIndex].systems[systemIndex].health = status;
        }
    }

    if (status == DCGM_HEALTH_RESULT_WARN)
    {
        PRINT_WARNING("%s", "%s", d.GetMessage().c_str());
    }
    else if (status == DCGM_HEALTH_RESULT_FAIL)
    {
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetPcie(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ADD_WATCH(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetMem(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ADD_WATCH(DCGM_FI_DEV_ECC_SBE_VOL_TOTAL);
    ADD_WATCH(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);

    // single and double bit retired pages
    // the sampling of 1 second is fine for the above, these however should have a longer sampling rate
    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_RETIRED_SBE, 
                                        30000000 /* 30 seconds */, 
                                        691200.0 /* 8 days of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_RETIRED_SBE << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }
    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_RETIRED_DBE, 
                                        30000000 /* 30 seconds */, 
                                        691200.0 /* 8 days of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_RETIRED_DBE << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }
    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_RETIRED_PENDING, 
                                        30000000 /* 30 seconds */, 
                                        691200.0 /* 8 days of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_RETIRED_PENDING << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetInforom(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_INFOROM_CONFIG_VALID, 
                                        3600000000 /* 1 hour */, 
                                        86400.0 /* 1 day of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_INFOROM_CONFIG_VALID << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }
 
    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetThermal(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_THERMAL_VIOLATION, 
                                        30000000 /* 30 seconds */, 
                                        86400.0 /* 1 day of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_THERMAL_VIOLATION << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetPower(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_POWER_VIOLATION, 
                                        30000000 /* 30 sec */, 
                                        86400.0 /* 1 day of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_POWER_VIOLATION << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }
    
    ret = mpCacheManager->AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_POWER_USAGE, 
                                        30000000 /* 30 sec */, 
                                        86400.0 /* 1 day of samples */, 
                                        0, watcher, false);
    if (DCGM_ST_OK != ret)
    {
        std::stringstream ss;
        ss << "Failed to set watch for field " << DCGM_FI_DEV_POWER_VIOLATION << " on GPU " << gpuId;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        return ret;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::SetLWLink(unsigned int gpuId, bool enable, DcgmWatcher watcher)
{
    // lwrrently if a watch is removed it removes for the entire system (i.e. no reference counter)
    // thus ignore the "enable" flag for now
    dcgmReturn_t ret = DCGM_ST_OK;

    if (!enable) //ignore
        return ret;

    ADD_WATCH(DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    ADD_WATCH(DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL);
    ADD_WATCH(DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL);
    ADD_WATCH(DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorPcie(dcgm_field_entity_group_t entityGroupId, 
                                          dcgm_field_eid_t entityId, long long startTime, 
                                          long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    unsigned short fieldId = DCGM_FI_DEV_PCIE_REPLAY_COUNTER;
    dcgmFieldValue_v1 startValue, endValue;

    int count = 0;
    unsigned int oneMinuteInUsec = 60000000;
    timelib64_t now = timelib_usecSince1970();
            

    /* Update the start and the end time if they are blank */    
    if(!startTime) {
        startTime = now - oneMinuteInUsec;
    }

    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    /* Get the value of the field at the StartTime*/
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField, 
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_ASCENDING, &startValue));

    if (DCGM_ST_NO_DATA == ret)
    {
        PRINT_DEBUG("%u", "No data for PCIe for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_NOT_WATCHED == ret)
    {
        PRINT_WARNING("%u", "PCIe not watched for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d %u", "fpdcgmGetMultipleValuesForField returned %d for gpuId %u", 
                    (int)ret, entityId);
        return ret;
    }

    if (DCGM_INT64_IS_BLANK(startValue.value.i64))
        return DCGM_ST_OK;

    /* Get the value of the field at the endTime*/    
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));
    if (DCGM_ST_NO_DATA == ret)
    {
        PRINT_DEBUG("%u", "No data for PCIe for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_NOT_WATCHED == ret)
    {
        PRINT_WARNING("%u", "PCIe not watched for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d %u", "fpdcgmGetMultipleValuesForField returned %d for gpuId %u", (int)ret, entityId);
        return ret;
    }

    if (DCGM_INT64_IS_BLANK(endValue.value.i64))
        return DCGM_ST_OK;


    // NO DATA is handled automatically so here we can assume we have the values from the last minute
    // both values have been checked for BLANK values so can be used here
    int pciReplayRate = (startValue.value.i64 >= endValue.value.i64)? (startValue.value.i64 - endValue.value.i64) : (endValue.value.i64 - startValue.value.i64);
    
    
    if (pciReplayRate > DCGM_LIMIT_MAX_PCIREPLAY_RATE)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCI_REPLAY_RATE, d, DCGM_LIMIT_MAX_PCIREPLAY_RATE, entityId,
                                  pciReplayRate);
        SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_PCIE, 
                    d, response, true);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string DcgmHealthWatch::MemFieldToString(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_ECC_SBE_VOL_TOTAL:
            return "Volatile SBEs";
        case DCGM_FI_DEV_ECC_DBE_VOL_TOTAL:
            return "Volatile DBEs";
        default:
            return "Error";
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorMem(dcgm_field_entity_group_t entityGroupId, 
                                         dcgm_field_eid_t entityId, long long startTime, 
                                         long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t localReturn = DCGM_ST_OK;
    dcgmFieldValue_v1 startValue, endValue; // for rates
    unsigned short fieldIds[DCGM_HEALTH_MEMORY_NUM_FIELDS] = {0};
    unsigned int oneMinuteInUsec = 60000000;
    timelib64_t now = timelib_usecSince1970();
    int count = 0;

    fieldIds[0] = DCGM_FI_DEV_ECC_SBE_VOL_TOTAL;
    fieldIds[1] = DCGM_FI_DEV_ECC_DBE_VOL_TOTAL;
    fieldIds[2] = DCGM_FI_DEV_RETIRED_SBE;
    fieldIds[3] = DCGM_FI_DEV_RETIRED_DBE;
    fieldIds[4] = DCGM_FI_DEV_RETIRED_PENDING;

    bool newIncident = true;

    /* Update the start and the end time if they are blank */    
    if(!startTime) {
        startTime = now - oneMinuteInUsec;
    }
    
    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    // first handle the actual error counts
    // if our stored value is greater than the returned value then someone likely
    // reset the volatile counter.  Just reset ours
    for (unsigned int counter = 0; counter < DCGM_HEALTH_MEMORY_LOC_COUNT; counter++)
    {
        count = 1;

        localReturn = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
                ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldIds[counter], &count, startTime,
                 endTime, DCGM_ORDER_DESCENDING, &endValue));

        if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA && localReturn != DCGM_ST_NOT_WATCHED)
            ret = localReturn;

        if(DCGM_INT64_IS_BLANK(endValue.value.i64))
        {
            PRINT_DEBUG("%d %u", "Skipping blank fieldId %d, index %u", fieldIds[counter], counter);
            continue;
        }

        if (fieldIds[counter] == DCGM_FI_DEV_ECC_DBE_VOL_TOTAL) /* DBE Total */
        {
            // Fail for any volatile DBEs
            if (endValue.value.i64 > 0)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_VOLATILE_DBE_DETECTED, d,
                                          static_cast<unsigned int>(endValue.value.i64), entityId);
                SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_MEM, 
                            d, response, newIncident);
                newIncident = false;
            }
        }
        else // SBE
        {
            // for SBE we are looking at a rate defined by DCGM_LIMIT_MAX_SBE_RATE
            count = 1;
            long long sbeRate = DCGM_INT64_BLANK;

            DcgmcmSummaryType_t summaryType = DcgmcmSummaryTypeDifference;
            localReturn = mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU, entityId, fieldIds[counter], 1, &summaryType, &sbeRate, startTime, endTime, 0, 0);

            if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA)
                ret = localReturn;

            PRINT_DEBUG("%d %u %lld", "localReturn %d, fieldId %u, sbeRate %lld", (int)localReturn, fieldIds[counter], sbeRate);

            if(DCGM_INT64_IS_BLANK(sbeRate))
                continue; /* No SBE samples or not supported */
            
            if(sbeRate < 0)
                sbeRate = -sbeRate; /* If the GPU was reset, this value may have reset to 0. take the absolute value */

            if (sbeRate >= DCGM_LIMIT_MAX_SBE_RATE)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_VOLATILE_SBE_DETECTED, d, DCGM_LIMIT_MAX_SBE_RATE,
                                          entityId, sbeRate);
                SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_MEM, 
                            d, response, newIncident);
                newIncident = false;
            }
        }
    }

    count = 1;
    localReturn = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldIds[DCGM_HEALTH_MEMORY_PAGE_RETIRED_PENDING], &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));

    if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA && localReturn != DCGM_ST_NOT_WATCHED)
        ret = localReturn;
    
    // now handle retired pages
    if (endValue.value.i64 && endValue.status == DCGM_ST_OK &&
        endValue.value.i64 != DCGM_INT64_NOT_SUPPORTED)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PENDING_PAGE_RETIREMENTS, d, entityId);
        SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_MEM, d, response,
                    newIncident);
        newIncident = false;
    }

    count = 1;
    dcgmFieldValue_v1 sbeRetiredPage, dbeRetiredPage;
    localReturn = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId,fieldIds[DCGM_HEALTH_MEMORY_PAGE_RETIRED_DBE], &count, startTime, endTime, DCGM_ORDER_DESCENDING, &dbeRetiredPage));

    if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA&& localReturn != DCGM_ST_NOT_WATCHED)
        ret = localReturn;
    
    count = 1;
    localReturn = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId,fieldIds[DCGM_HEALTH_MEMORY_PAGE_RETIRED_SBE], &count, startTime, endTime, DCGM_ORDER_DESCENDING, &sbeRetiredPage));

    if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA&& localReturn != DCGM_ST_NOT_WATCHED)
        ret = localReturn;        

    if (sbeRetiredPage.status == DCGM_ST_OK && dbeRetiredPage.status == DCGM_ST_OK &&
        sbeRetiredPage.value.i64 != DCGM_INT64_NOT_SUPPORTED &&
         dbeRetiredPage.value.i64 != DCGM_INT64_NOT_SUPPORTED)
    {
        // the combined total of retired pages should not be more than or equal to DCGM_LIMIT_MAX_RETIRED_PAGES
        // which is set via bug 1665722
        if ((dbeRetiredPage.value.i64 + sbeRetiredPage.value.i64) >= DCGM_LIMIT_MAX_RETIRED_PAGES)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_RETIRED_PAGES_LIMIT, d, DCGM_LIMIT_MAX_RETIRED_PAGES, entityId);
            SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_MEM, d, response, 
                        newIncident);
            newIncident = false;
            return ret;
        }

        // The dbe retired pages should not be more than DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT
        // *AND* be aclwmulating more than 1 per week after the limit has been met
        // JIRA DCGM-458
        if (dbeRetiredPage.value.i64 > DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT) 
        {
            // Check whether the rate of continuing page retirments (after the SOFT_LIMIT) meets the failure condition.
            dcgmReturn_t localReturn = DCGM_ST_OK;
            dcgmFieldValue_v1 oneWeekAgoDbeRetiredPages;
            timelib64_t oneWeekInUsec = 604800000000;
            int count = 1;
            timelib64_t now = timelib_usecSince1970();
            // Get the number of dbe retired pages before current week
            localReturn = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
                    ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldIds[DCGM_HEALTH_MEMORY_PAGE_RETIRED_DBE], 
                    &count, 0, now - oneWeekInUsec, DCGM_ORDER_DESCENDING, &oneWeekAgoDbeRetiredPages));
            
            if (localReturn != DCGM_ST_OK && localReturn != DCGM_ST_NO_DATA)
            {
                ret = localReturn;
            }
            
            int64_t dbePagesRetiredThisWeek = dbeRetiredPage.value.i64 - oneWeekAgoDbeRetiredPages.value.i64;
            if (dbePagesRetiredThisWeek > 1)
            {
                // More than one page retired due to DBE in the past week, failure condition met.
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_RETIRED_PAGES_DBE_LIMIT, d,
                                          DCGM_LIMIT_MAX_RETIRED_PAGES_SOFT_LIMIT, entityId);
                SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_MEM, d, response, 
                            newIncident);
                newIncident = false;
            }
        }
    }

    return ret;
}

dcgmReturn_t DcgmHealthWatch::MonitorInforom(dcgm_field_entity_group_t entityGroupId, 
                                             dcgm_field_eid_t entityId, long long startTime, 
                                             long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmFieldValue_v1 endValue;
    unsigned short fieldId = DCGM_FI_DEV_INFOROM_CONFIG_VALID;
    timelib64_t now = timelib_usecSince1970();
    unsigned int oneMinuteInUsec = 60000000;
    int count = 0;

    if(!startTime) {
        startTime = 0; /* Check from the start of the cache */
    }

    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    /* check for the fieldValue at the endTime*/
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
        ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));

    if (DCGM_ST_NO_DATA == ret)
    {
        PRINT_DEBUG("%u", "No data for inforom for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_NOT_WATCHED == ret)
    {
        PRINT_WARNING("%u", "Not watched for inforom for gpuId %u", entityId);
        return DCGM_ST_OK;
    }
    else if (DCGM_ST_OK != ret)
    {
        PRINT_ERROR("%d %u", "Unable to retrieve field %d from cache. gpuId %u", fieldId, entityId);
        return ret;
    }

    if (DCGM_INT64_IS_BLANK(endValue.value.i64))
        return DCGM_ST_OK;
    
    if (!(endValue.value.i64))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CORRUPT_INFOROM, d, entityId);
        SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_INFOROM, d, response,
                    true);
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorThermal(dcgm_field_entity_group_t entityGroupId, 
                                             dcgm_field_eid_t entityId, long long startTime, 
                                             long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    unsigned short fieldId = DCGM_FI_DEV_THERMAL_VIOLATION;
    dcgmFieldValue_v1 startValue, endValue;
    int count = 0;
    long long int violationTime = 0;

    timelib64_t now = timelib_usecSince1970();
    unsigned int oneMinuteInUsec = 60000000;

    /* Update the start and the end time if they are blank */    
    if(!startTime) {
        startTime = now - oneMinuteInUsec;
    }
    
    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    /* Get the value at the startTime */
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField, 
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_ASCENDING, &startValue));

    if (DCGM_ST_NO_DATA == ret)
        return DCGM_ST_OK;
    if (DCGM_ST_OK != ret)
        return ret;
    if (DCGM_INT64_IS_BLANK(startValue.value.i64))
        return DCGM_ST_OK;


    /* Get the value at the endTime*/
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));
    
    if (DCGM_ST_NO_DATA == ret)
        return DCGM_ST_OK;
    if (DCGM_ST_OK != ret)
        return ret;
    if (DCGM_INT64_IS_BLANK(endValue.value.i64))
        return DCGM_ST_OK;

    // NO DATA is handled automatically so here we can assume we have the values from the last minute
    // both values have been checked for BLANK values so can be used here
    violationTime = startValue.value.i64 >= endValue.value.i64 ? (startValue.value.i64 - endValue.value.i64) : (endValue.value.i64 - startValue.value.i64) ;

    if (violationTime)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CLOCK_THROTTLE_THERMAL, d, entityId);
        SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_THERMAL, d,
                    response, true);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorPower(dcgm_field_entity_group_t entityGroupId, 
                                           dcgm_field_eid_t entityId, long long startTime, 
                                           long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    unsigned short fieldId = DCGM_FI_DEV_POWER_VIOLATION;
    dcgmFieldValue_v1 lwrrentValue, pastValue;
    dcgmFieldValue_v1 startValue, endValue;
    unsigned int oneMinuteInUsec = 60000000;
    int count = 0;
    long long int violationTime = 0;
    dcgmcm_sample_t sample = { 0 };
    bool firstFailure = true;

    timelib64_t now = timelib_usecSince1970();

    // Warn if we cannot read the power on this GPU
    if (entityGroupId == DCGM_FE_GPU)
    {
        ret = mpCacheManager->GetLatestSample(DCGM_FE_GPU, entityId, DCGM_FI_DEV_POWER_USAGE, &sample, 0);
        if (ret == DCGM_ST_OK && DCGM_FP64_IS_BLANK(sample.val.d) && sample.val.d != DCGM_FP64_NOT_SUPPORTED)
        {
            // We aren't successfully reading the power for this GPU, add a warning
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_POWER_UNREADABLE, d, entityId);
            SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, DCGM_HEALTH_WATCH_POWER, d,
                        response, firstFailure);
            firstFailure = false;
        }
    }

    /* Update the start and the end time if they are blank */    
    if (!startTime)
    {
        startTime = now - oneMinuteInUsec;
    }
    
    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    /* Update the value at the start time*/
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField, 
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_ASCENDING, &startValue));

    if (DCGM_ST_NO_DATA == ret)
        return DCGM_ST_OK;
    if (DCGM_ST_OK != ret)
        return ret;
    if (DCGM_INT64_IS_BLANK(startValue.value.i64))
        return DCGM_ST_OK;


    /* Update the value at the end time */
    count = 1;
    ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldId, &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));
    if (DCGM_ST_NO_DATA == ret)
        return DCGM_ST_OK;
    if (DCGM_ST_OK != ret)
        return ret;
    if (DCGM_INT64_IS_BLANK(endValue.value.i64))
        return DCGM_ST_OK;

    // NO DATA is handled automatically so here we can assume we have the values from the last minute
    // both values have been checked for BLANK values so can be used here
    violationTime = startValue.value.i64 >= endValue.value.i64 ? (startValue.value.i64 - endValue.value.i64) : (endValue.value.i64 - startValue.value.i64) ;

    if (violationTime)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CLOCK_THROTTLE_POWER, d, entityId);
        SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_WARN, 
                    DCGM_HEALTH_WATCH_POWER, d, response, firstFailure);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorLWLink(dcgm_field_entity_group_t entityGroupId, 
                                            dcgm_field_eid_t entityId, 
                                            long long startTime, 
                                            long long endTime, void *response)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    unsigned short fieldIds[DCGM_HEALTH_WATCH_LWLINK_ERROR_NUM_FIELDS] = {0};
    dcgmFieldValue_v1 startValue, endValue;
    int count = 0;

    /* Various LWLink error counters to be monitored */
    fieldIds[0] = DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL;
    fieldIds[1] = DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL;
    fieldIds[2] = DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL;
    fieldIds[3] = DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL;

    unsigned int oneMinuteInUsec = 60000000;
    timelib64_t now = timelib_usecSince1970();

    /* Update the start and the end time if they are blank */    
    if(!startTime) {
        startTime = now - oneMinuteInUsec;
    }
    
    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    for(unsigned int lwLinkField = 0; lwLinkField < DCGM_HEALTH_WATCH_LWLINK_ERROR_NUM_FIELDS ; lwLinkField++)
    {
        count = 1;
        ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldIds[lwLinkField], &count, startTime, endTime, DCGM_ORDER_ASCENDING, &startValue));

        if(ret != DCGM_ST_OK && ret != DCGM_ST_NO_DATA && ret !=DCGM_ST_NOT_WATCHED)
            return ret;
        
        /* If the field is not supported, continue with others */
        if(ret == DCGM_ST_NO_DATA || startValue.value.i64 ==DCGM_INT64_NOT_SUPPORTED || DCGM_INT64_IS_BLANK(startValue.value.i64))
            continue;

        count = 1;
        ret = DCGM_CALL_ETBL(m_etblLwcmEngineInternal, fpdcgmGetMultipleValuesForField,
            ((dcgmHandle_t)DCGM_EMBEDDED_HANDLE, entityId, fieldIds[lwLinkField], &count, startTime, endTime, DCGM_ORDER_DESCENDING, &endValue));

        if(ret != DCGM_ST_OK && ret != DCGM_ST_NO_DATA)
            return ret;

        /* Continue with other fields if this value is BLANK or has no data  */
        if (ret == DCGM_ST_NO_DATA || DCGM_INT64_IS_BLANK(endValue.value.i64))
            continue;

        // NO DATA is handled automatically so here we can assume we have the values from the last minute
        // both values have been checked for BLANK values so can be used here

        int64_t lwLinkError = (startValue.value.i64 >= endValue.value.i64) ? (startValue.value.i64 - endValue.value.i64) : (endValue.value.i64 - startValue.value.i64);

        if (lwLinkError >= DCGM_LIMIT_MAX_LWLINK_ERROR)
        {
            dcgm_field_meta_p fm = DcgmFieldGetById(fieldIds[lwLinkField]);
            char fieldTag[128];
            dcgmHealthWatchResults_t res = DCGM_HEALTH_RESULT_WARN;
            DcgmError d;

            if (fm != NULL)
            {
                snprintf(fieldTag, sizeof(fieldTag), "%s", fm->tag);
            }
            else
            {
                snprintf(fieldTag, sizeof(fieldTag), "Unknown field %hu", fieldIds[lwLinkField]);
            }


            if ((fieldIds[lwLinkField] == DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL) ||
                (fieldIds[lwLinkField] == DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL))
            {
                // Replay and recovery errors are failures, not warnings.
                res = DCGM_HEALTH_RESULT_FAIL;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWLINK_ERROR_CRITICAL, d, lwLinkError, fieldTag, entityId);
            }
            else
            {
                // CRC errors are only an error if more than 100 are happening per second
                double timeDiffInSec;
                if (endTime == 0)
                {
                    // Use now as the end time
                    timeDiffInSec = (now - startTime) / 1000000.0;
                }
                else
                {
                    timeDiffInSec = (endTime - startTime) / 1000000.0;
                }
                double perSec = static_cast<double>(lwLinkError) / timeDiffInSec;
                if (perSec >= DCGM_LIMIT_MAX_LWLINK_CRC_ERROR)
                {
                    res = DCGM_HEALTH_RESULT_FAIL;
                    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWLINK_CRC_ERROR_THRESHOLD, d, perSec, fieldTag, entityId);
                }
                else
                {
                    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWLINK_ERROR_THRESHOLD, d, lwLinkError, fieldTag, entityId,
                                              DCGM_LIMIT_MAX_LWLINK_ERROR);
                }
            }

            SetResponse(entityGroupId, entityId, res, DCGM_HEALTH_WATCH_LWLINK, d, response, true);
        }
    }


    /* See if any links are down */
    dcgmLwLinkLinkState_t linkStates[DCGM_LWLINK_MAX_LINKS_PER_GPU];
    ret = mpCacheManager->GetEntityLwLinkLinkStatus(DCGM_FE_GPU, entityId, linkStates);
    if(ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %u", "Got error %d from GetEntityLwLinkLinkStatus gpuId %u", 
                    (int)ret, entityId);
        return ret;
    }
    for(int i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_GPU; i++)
    {
        if(linkStates[i] == DcgmLwLinkLinkStateDown)
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWLINK_DOWN, d, entityId, i);
            SetResponse(entityGroupId, entityId, DCGM_HEALTH_RESULT_FAIL, DCGM_HEALTH_WATCH_LWLINK, d, response,
                        true);
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHealthWatch::MonitorLwSwitchErrorCounts(bool fatal, 
                                                         dcgm_field_entity_group_t entityGroupId, 
                                                         dcgm_field_eid_t entityId, 
                                                         long long startTime, 
                                                         long long endTime, void *response)
{
    std::vector<unsigned int>::iterator fieldIdIter;
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int> *fieldIds;
    dcgmcm_sample_t sample;
    dcgmHealthWatchResults_t healthWatchResult;
    dcgmHealthSystems_t healthWatchSystems;
    std::string errorTypeString;

    unsigned int oneMinuteInUsec = 60000000;
    timelib64_t now = timelib_usecSince1970();

    /* Update the start and the end time if they are blank */    
    if(!startTime) 
    {
        startTime = now - oneMinuteInUsec;
    }
    
    /* Note: Allow endTime to be in the future. 0 = blank = most recent record in time series */

    if(fatal)
    {
        fieldIds = &m_lwSwitchFatalFieldIds;
        healthWatchResult = DCGM_HEALTH_RESULT_FAIL;
        healthWatchSystems = DCGM_HEALTH_WATCH_LWSWITCH_FATAL;
        errorTypeString = "fatal";
    }
    else /* Non-fatal */
    {
        fieldIds = &m_lwSwitchNonFatalFieldIds;
        healthWatchResult = DCGM_HEALTH_RESULT_WARN;
        healthWatchSystems = DCGM_HEALTH_WATCH_LWSWITCH_NONFATAL;
        errorTypeString = "nonfatal";
    }

    memset(&sample, 0, sizeof(sample));

    for(fieldIdIter = fieldIds->begin(); fieldIdIter != fieldIds->end(); ++fieldIdIter)
    {
        int count = 1;
        dcgmReturn = mpCacheManager->GetSamples(entityGroupId, entityId, *fieldIdIter, 
                                                &sample, &count, startTime, endTime, 
                                                DCGM_ORDER_DESCENDING);
        if(dcgmReturn != DCGM_ST_OK || !count)
        {
            PRINT_DEBUG("%d %u %u %u %lld %lld", "return %d for GetSamples eg %u, eid %u, "
                        "fieldId %u, start %lld, end %lld", (int)dcgmReturn,
                        entityGroupId, entityId, *fieldIdIter, startTime, endTime);
            continue;
        }

        if(sample.val.i64 > 0)
        {
            unsigned int linkId = (*fieldIdIter) - fieldIds->at(0);
            DcgmError d;
            if (fatal)
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWSWITCH_FATAL_ERROR, d, entityId, linkId);
            }
            else
            {
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWSWITCH_NON_FATAL_ERROR, d, entityId, linkId);
            }

            SetResponse(entityGroupId, entityId, healthWatchResult, healthWatchSystems, d, response, true);
        }
    }

    /* See if any links are down. Only do this for the fatal case so we don't get duplicate errors for both fatal and non-fatal */
    if(fatal)
    {
        dcgmLwLinkLinkState_t linkStates[DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH];
        dcgmReturn = mpCacheManager->GetEntityLwLinkLinkStatus(DCGM_FE_SWITCH, entityId, linkStates);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u", "Got error %d from GetEntityLwLinkLinkStatus eid %u", 
                        (int)dcgmReturn, entityId);
            return dcgmReturn;
        }
        for(int i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; i++)
        {
            if(linkStates[i] == DcgmLwLinkLinkStateDown)
            {
                DcgmError d;
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LWLINK_DOWN, d, entityId, i);
                SetResponse(entityGroupId, entityId, healthWatchResult, healthWatchSystems, d, response, true);
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmHealthWatch::OnGroupRemove(unsigned int groupId)
{
    groupWatchTable_t::iterator groupWatchIter;

    dcgm_mutex_lock(m_mutex);

    groupWatchIter = mGroupWatchState.find(groupId);
    if(groupWatchIter == mGroupWatchState.end())
    {
        PRINT_DEBUG("%u", "OnGroupRemove didn't find groupId %u", groupId);
    }
    else
    {
        mGroupWatchState.erase(groupWatchIter);
        PRINT_DEBUG("%u", "OnGroupRemove found and removed groupId %u", groupId);
    }

    dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/

