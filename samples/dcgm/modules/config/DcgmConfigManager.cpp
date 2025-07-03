/* 
 * File:   DcgmConfigManager.cpp
 */

#include "DcgmConfigManager.h"
#include "lwcmvalue.h"
#include <sstream>

#include "logging.h"
#include "lwos.h"
#include "lwml.h"

DcgmConfigManager::DcgmConfigManager(DcgmCacheManager *pCacheManager, LwcmGroupManager *pGroupManager) 
{
    mpCacheManager = pCacheManager;
    mpGroupManager = pGroupManager;
    mClocksConfigured = 0;
    
    m_mutex = new DcgmMutex(0);

    memset(m_activeConfig, 0, sizeof(m_activeConfig));
}

/*****************************************************************************/
bool DcgmConfigManager::RunningAsRoot(void)
{
    if(geteuid() == 0)
        return true;
    else
        return false;
}

/*****************************************************************************/
DcgmConfigManager::~DcgmConfigManager() 
{
    int i;

    /* Cleanup Data structures */
    dcgm_mutex_lock(m_mutex);
        
    for(i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if(m_activeConfig[i])
        {
            free(m_activeConfig[i]);
            m_activeConfig[i] = 0;
        }
    }
    
    dcgm_mutex_unlock(m_mutex);
    
    delete m_mutex;
    m_mutex = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetEccMode(unsigned int gpuId, dcgmConfig_t *setConfig,
                                                 dcgmConfig_t *lwrrentConfig, bool *pIsResetNeeded)
{
    dcgmReturn_t dcgmRet;

    if(DCGM_INT32_IS_BLANK(setConfig->eccMode))
    {
        PRINT_DEBUG("", "ECC mode was blank");
        return DCGM_ST_OK;
    }

    /* Is ECC even supported by the hardware? */
    if(DCGM_INT32_IS_BLANK(lwrrentConfig->eccMode))
    {
        PRINT_DEBUG("%u", "ECC mode was blank for gpuId %u", gpuId);
        return DCGM_ST_OK;
    }

    if(lwrrentConfig->eccMode == setConfig->eccMode)
    {
        PRINT_DEBUG("%u %u", "ECC mode %u already matches for gpuId %u.", 
                    setConfig->eccMode, gpuId);
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t valueToSet;

    memset(&valueToSet, 0, sizeof(valueToSet));
    valueToSet.val.i64 = setConfig->eccMode;

    dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_ECC_PENDING, &valueToSet);
    if(dcgmRet != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %u %u", "Got error %d while setting ECC to %u for gpuId %u",
                    dcgmRet, setConfig->eccMode, gpuId);
        return dcgmRet;
    }

    *pIsResetNeeded = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetPowerLimit(unsigned int gpuId, 
                                                    dcgmConfig_t *setConfig)
{
    dcgmReturn_t dcgmRet;

    if(DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
    {
        PRINT_DEBUG("%u", "Power limit was blank for gpuId %u", gpuId);
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t value;
    memset(&value, 0, sizeof(value));
    value.val.d = setConfig->powerLimit.val;

    dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, &value);
    if (DCGM_ST_OK != dcgmRet) 
    {
        PRINT_ERROR("%d %d", "Error in setting power limit for GPU ID: %d Error: %d", 
                    gpuId, (int)dcgmRet);
        return dcgmRet;
    }
    
    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetPerfState(unsigned int gpuId, dcgmConfig_t *setConfig)
{
    dcgmReturn_t dcgmRet;
    unsigned int targetMemClock, targetSmClock;
    dcgmcm_sample_t value;

    targetMemClock = setConfig->perfState.targetClocks.memClock;
    targetSmClock = setConfig->perfState.targetClocks.smClock;

    if(DCGM_INT32_IS_BLANK(targetMemClock) && DCGM_INT32_IS_BLANK(targetSmClock)) 
    {
        PRINT_DEBUG("%u", "Both memClock and smClock were blank for gpuId %u", gpuId);
        /* Ignore the clock settings if both clock values are BLANK */
        return DCGM_ST_OK;
    }

    /* Update the Clock Configured to 1 */
    mClocksConfigured = 1;
    
    /* Are both 0s? That means reset target clocks */
    if(targetMemClock == 0  && targetSmClock == 0)
    {
        memset(&value, 0, sizeof(value));
        value.val.i64 = lwcmvalue_int32_to_int64(targetMemClock);
        value.val2.i64 = lwcmvalue_int32_to_int64(targetSmClock);

        /* Set the clock. 0-0 implies Reset */
        dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, &value);
        if (DCGM_ST_OK != dcgmRet)
        {
            PRINT_ERROR("%d %d %d %d", "Can't set fixed clocks %d,%d for GPU Id %d. Error: %d",
                        targetMemClock, targetSmClock, gpuId, dcgmRet);
            return dcgmRet;
        }

        /* Reenable auto boosted clocks when app clocks are disabled */
        memset(&value, 0, sizeof(value));
        value.val.i64 = 1;  // Enable Auto Boost Mode
        dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_AUTOBOOST, &value);
        if (dcgmRet == DCGM_ST_NOT_SUPPORTED)
        {
            /* Not an error for >= Pascal. LWML returns NotSupported */
            PRINT_DEBUG("%d", "Got NOT_SUPPORTED when setting auto boost for gpuId %d", gpuId);
            /* Return success below */
        }
        else if (DCGM_ST_OK != dcgmRet)
        {
            PRINT_ERROR("%d %d", "Can't set Auto-boost for GPU Id %d. Error: %d", gpuId, dcgmRet);
            return dcgmRet;
        }

        return DCGM_ST_OK;
    }

    /* Set the clock */
    memset(&value, 0, sizeof(value));
    value.val.i64 = lwcmvalue_int32_to_int64(targetMemClock);
    value.val2.i64 = lwcmvalue_int32_to_int64(targetSmClock);
    dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, &value);
    if (DCGM_ST_OK != dcgmRet)
    {
        PRINT_ERROR("%d %d %d %d", "Can't set fixed clocks %d,%d for GPU Id %d. Error: %d",
                    targetMemClock, targetSmClock, gpuId, dcgmRet);
        return dcgmRet;
    }

    /* Disable auto boosted clocks when app clocks are set */
    memset(&value, 0, sizeof(value));
    value.val.i64 = 0;  // Disable Auto Boost Mode
    dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_AUTOBOOST, &value);
    if (dcgmRet == DCGM_ST_NOT_SUPPORTED)
    {
        /* Not an error for >= Pascal. LWML returns NotSupported */
        PRINT_DEBUG("%d", "Got NOT_SUPPORTED when setting auto boost for gpuId %d", gpuId);
        /* Return success below */
    }
    else if (DCGM_ST_OK != dcgmRet)
    {
        PRINT_ERROR("%d %d", "Can't set Auto-boost for GPU Id %d. Error: %d", gpuId, dcgmRet);
        return dcgmRet;
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetComputeMode(unsigned int gpuId, dcgmConfig_t *config)
{
    dcgmReturn_t dcgmRet;

    if (DCGM_INT32_IS_BLANK(config->computeMode))
    {
        PRINT_DEBUG("", "compute mode was blank");
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t value;
    memset(&value, 0, sizeof(value));
    value.val.i64 = config->computeMode;

    dcgmRet = mpCacheManager->SetValue(gpuId, DCGM_FI_DEV_COMPUTE_MODE, &value);
    if (DCGM_ST_OK != dcgmRet) 
    {
        PRINT_ERROR("%d %d", "Failed to set compute mode for GPU ID: %d Error: %d", gpuId, dcgmRet);
        return dcgmRet;
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
static void dcmBlankConfig(dcgmConfig_t *config, unsigned int gpuId)
{
    /* Make a blank record */
    memset(config, 0, sizeof(*config));
    config->version = dcgmConfig_version;
    config->gpuId = gpuId;
    config->eccMode = DCGM_INT32_BLANK;
    config->computeMode = DCGM_INT32_BLANK;
    config->perfState.syncBoost = DCGM_INT32_BLANK;
    config->perfState.targetClocks.version = dcgmClockSet_version;
    config->perfState.targetClocks.memClock = DCGM_INT32_BLANK;
    config->perfState.targetClocks.smClock = DCGM_INT32_BLANK;
    config->powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
    config->powerLimit.val = DCGM_INT32_BLANK;
}

/*****************************************************************************/
dcgmConfig_t *DcgmConfigManager::HelperGetTargetConfig(unsigned int gpuId)
{
    dcgmConfig_t *retVal = 0;

    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);
    
    retVal = m_activeConfig[gpuId];

    if(!retVal)
    {
        retVal = (dcgmConfig_t *)malloc(sizeof(dcgmConfig_t));
        dcmBlankConfig(retVal, gpuId);

        /* Activate our blank record */
        m_activeConfig[gpuId] = retVal;
    }

    if(mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    return retVal;
}

/*****************************************************************************/
void DcgmConfigManager::HelperMergeTargetConfiguration(unsigned int gpuId, unsigned int fieldId, 
                                                       dcgmConfig_t *setConfig)
{
    dcgmConfig_t *targetConfig = HelperGetTargetConfig(gpuId);
    if(targetConfig == setConfig)
    {
        PRINT_WARNING("", "Caller tried to set targetConfig to identical setConfig.");
        return;
    }

    switch (fieldId)
    {
        case DCGM_FI_DEV_ECC_LWRRENT:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->eccMode))
                targetConfig->eccMode = setConfig->eccMode;
            break;
        }
            
        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
                targetConfig->powerLimit.val = setConfig->powerLimit.val;
            break;
        }
            
        case DCGM_FI_DEV_APP_SM_CLOCK:  /* Fall-through is intentional */
        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.targetClocks.memClock))
                targetConfig->perfState.targetClocks.memClock = setConfig->perfState.targetClocks.memClock;
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.targetClocks.smClock))
                targetConfig->perfState.targetClocks.smClock = setConfig->perfState.targetClocks.smClock;
            break;
        }
            
        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->computeMode))
                targetConfig->computeMode = setConfig->computeMode;
            
            break;
        }
            
        case DCGM_FI_SYNC_BOOST:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.syncBoost))
                targetConfig->perfState.syncBoost = setConfig->perfState.syncBoost;
            break;
        }

        default:
            PRINT_ERROR("%u", "Unhandled fieldId %u", fieldId);
            // Should never happen
            break;
            
    }
    
    return;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetConfigGpu(unsigned int gpuId, 
                                             dcgmConfig_t *setConfig, 
                                             DcgmConfigManagerStatusList *statusList)
{
    unsigned int multiPropertyRetCode = 0;
    dcgmReturn_t dcgmRet;
    dcgmConfig_t lwrrentConfig;

    if(!setConfig)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    if(setConfig->version != dcgmConfig_version)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_VER_MISMATCH);
        return DCGM_ST_VER_MISMATCH;
    }

    if(!RunningAsRoot())
    {
        PRINT_DEBUG("", "SetConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    /* Most of the children of this function need the current config. Get it once */
    dcgmRet = GetLwrrentConfigGpu(gpuId, &lwrrentConfig);
    if(dcgmRet != DCGM_ST_OK)
    {
        PRINT_ERROR("%d %u", "Error %d from GetLwrrentConfigGpu() of gpuId %u", 
                    dcgmRet, gpuId);
        return dcgmRet;
    }

    /* Set Ecc Mode */
    bool isResetNeeded = false;
    dcgmRet = HelperSetEccMode(gpuId, setConfig, &lwrrentConfig, &isResetNeeded);
    if (DCGM_ST_OK != dcgmRet) 
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_LWRRENT, dcgmRet);
 
        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_ECC_LWRRENT, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_ECC_LWRRENT, setConfig);
    }
    
    /* Check if GPU reset is needed after GPU reset */
    if (isResetNeeded) 
    {
        PRINT_INFO("%d", "Reset Needed for GPU ID: %d", gpuId);

        /* Best effort to enforce the config */
        HelperEnforceConfig(gpuId, statusList);
    }
    
    /* Set Power Limit */
    dcgmRet = HelperSetPowerLimit(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet) {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, dcgmRet);
        
        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, setConfig);
    } else {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, setConfig);
    }
    
    /* Set Perf States */
    dcgmRet = HelperSetPerfState(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet) 
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, dcgmRet);
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, setConfig);
        
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, setConfig);
    }
    
    dcgmRet = HelperSetComputeMode(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_COMPUTE_MODE, dcgmRet);
        
        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_COMPUTE_MODE, setConfig);
        
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_COMPUTE_MODE, setConfig);
    }
    
    /* If any of the operation failed. Return it as an generic error */
    if (0 != multiPropertyRetCode) 
        return DCGM_ST_GENERIC_ERROR;
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetLwrrentConfigGpu(unsigned int gpuId, dcgmConfig_t *config)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    DcgmFvBuffer fvBuffer;
    dcgmReturn_t dcgmReturn;
    
    /* Blank out the values before we populate them */
    dcmBlankConfig(config, gpuId);

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId = gpuId;
    entities.push_back(entityPair);

    fieldIds.push_back(DCGM_FI_DEV_ECC_LWRRENT);
    fieldIds.push_back(DCGM_FI_DEV_APP_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_APP_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_POWER_MGMT_LIMIT);
    fieldIds.push_back(DCGM_FI_DEV_COMPUTE_MODE);

    dcgmReturn = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got error %d from GetMultipleLatestLiveSamples()", dcgmReturn);
        return dcgmReturn;
    }

    config->gpuId = gpuId;

    dcgmBufferedFvLwrsor_t cursor = 0;
    dcgmBufferedFv_t *fv;
    for(fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        if(fv->status != DCGM_ST_OK)
        {
            PRINT_DEBUG("%u %u %d", "Ignoring gpuId %u fieldId %u with status %d", 
                        fv->entityId, fv->fieldId, fv->status);
            continue;
        }

        switch(fv->fieldId)
        {
            case DCGM_FI_DEV_ECC_LWRRENT:
                config->eccMode = lwcmvalue_int64_to_int32(fv->value.i64);
                break;
            
            case DCGM_FI_DEV_APP_MEM_CLOCK:
                config->perfState.targetClocks.memClock = lwcmvalue_int64_to_int32(fv->value.i64);
                break;
            
            case DCGM_FI_DEV_APP_SM_CLOCK:
                config->perfState.targetClocks.smClock = lwcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT:
                config->powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
                config->powerLimit.val = lwcmvalue_double_to_int32(fv->value.dbl);
                break;
            
            case DCGM_FI_DEV_COMPUTE_MODE:
                config->computeMode = lwcmvalue_int64_to_int32(fv->value.i64);
                break;
            
            default:
                PRINT_ERROR("%u", "Unexpected fieldId %u", fv->fieldId);
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetLwrrentConfig(unsigned int groupId, 
                                                 unsigned int *numConfigs, 
                                                 dcgmConfig_t *configs, 
                                                 DcgmConfigManagerStatusList *statusList)
{
    int i;
    dcgmReturn_t dcgmReturn;
    unsigned int multiRetCode = 0;
    std::vector<unsigned int>gpuIds;

    if (!numConfigs || !configs || !statusList)
    {
        return DCGM_ST_BADPARAM;
    }
    *numConfigs = 0;

    if(!RunningAsRoot())
    {
        PRINT_DEBUG("", "GetLwrrentConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    /* Get group's gpu ids */
    dcgmReturn = mpGroupManager->GetGroupGpuIds(0, groupId, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Implies Invalid group ID */
        PRINT_ERROR("%d", "Config Get Err: Cannot get group Info from group Id:%d", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }
    
    /* Get number of gpus from the group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        PRINT_ERROR("%d", "Config Get Err: No GPUs configured for the group id : %d", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    for (i = 0; i < (int)gpuIds.size(); i++)
    {
        unsigned int gpuId;

        gpuId = gpuIds[i];

        dcgmReturn = GetLwrrentConfigGpu(gpuId, &configs[*numConfigs]);
        if(dcgmReturn != DCGM_ST_OK)
            multiRetCode++;
        
        (*numConfigs)++;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);
    
    /*
     * Special handling for sync boost on the group
     */
    unsigned int syncBoostIds[DCGM_MAX_NUM_DEVICES];

    dcgmReturn = HelperGetSyncBoostInfo(syncBoostIds);
    if (DCGM_ST_OK != dcgmReturn) 
    {
        multiRetCode++;
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, dcgmReturn);
    } 
    else 
    {
        unsigned int i;
        unsigned int gpuId;
        unsigned int syncBoostForGrp = 1; /* Start with sync boost flag as enabled */
        unsigned int rmSyncId;
        
        gpuId = gpuIds[0];
        
        /* Go through the list of GPUs to figure out if all the GPUs in the group are part of
            same sync boost ID */
        rmSyncId = syncBoostIds[gpuId];
        if (DCGM_INT32_NOT_FOUND != rmSyncId) 
        {
            for (i = 1; i < gpuIds.size(); i++) 
            {
                gpuId = gpuIds[i];
                if (rmSyncId != syncBoostIds[gpuId]) 
                {
                    syncBoostForGrp = 0;
                }
            }
        } 
        else 
        {
            syncBoostForGrp = 0;
        }
        
        /* Set all GPUs to have the same common syncboost state */
        for (i = 0; i < (*numConfigs); i++)
        {
            configs[i].perfState.syncBoost = syncBoostForGrp;
        }
    }

    if (multiRetCode != 0) 
        return DCGM_ST_GENERIC_ERROR;
    else
        return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetTargetConfig(unsigned int groupId, 
                                                unsigned int *numConfigs, 
                                                dcgmConfig_t *configs, 
                                                DcgmConfigManagerStatusList *statusList) 
{
    unsigned int index;
    dcgmReturn_t multiRetCode;
    std::vector<unsigned int>gpuIds;
    dcgmReturn_t dcgmReturn;

    if(!RunningAsRoot())
    {
        PRINT_DEBUG("", "GetTargetConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    dcgmReturn = mpGroupManager->GetGroupGpuIds(0, groupId, gpuIds);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetAllGroupIds", (int)dcgmReturn);
                    statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return dcgmReturn;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    /* Get number of gpus from thr group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        PRINT_ERROR("%d", "Config Get Err: No GPUs configured for the group id : %d", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }
    
    multiRetCode = DCGM_ST_OK;

    *numConfigs = 0;
    for (index = 0; index < gpuIds.size(); index++) 
    {
        unsigned int gpuId = gpuIds[index];
        dcgmConfig_t *activeConfig = HelperGetTargetConfig(gpuId);

        if(!activeConfig)
        {
            PRINT_ERROR("%u", "Unexpected NULL config for gpuId %u. OOM?", gpuId);
            statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_MEMORY);
            multiRetCode = DCGM_ST_MEMORY;
            continue;
        }

        memcpy(&configs[*numConfigs], activeConfig, sizeof(configs[0]));
        (*numConfigs)++;
    }
    
    return multiRetCode;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperEnforceConfig(unsigned int gpuId, 
                                                    DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmReturn; 
    unsigned int multiPropertyRetCode = 0;

    /* 
        activeConfig - the config that a user has set previously that we're going to enforce
        lwrrentConfig - the current state of the GPUs from the cache manager
    */
    dcgmConfig_t *activeConfig = m_activeConfig[gpuId];
    if(!activeConfig)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_NOT_CONFIGURED);
        return DCGM_ST_NOT_CONFIGURED;
    }

    dcgmConfig_t lwrrentConfig;
    dcgmReturn = GetLwrrentConfigGpu(gpuId, &lwrrentConfig);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%u %d", "Unable to get the current configuration for gpuId %u. st %d", 
                    gpuId, dcgmReturn);
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, dcgmReturn);
        return DCGM_ST_GENERIC_ERROR;
    }
    
    /* Set Ecc Mode */
    /* Always keep setting ECC mode as first. (might trigger GPU reset) */
    bool isResetNeeded = false;
    dcgmReturn = HelperSetEccMode(gpuId, activeConfig, &lwrrentConfig, &isResetNeeded);
    if (DCGM_ST_OK != dcgmReturn) 
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_LWRRENT, dcgmReturn);
    }

    if (isResetNeeded) 
    {
        dcgmReturn = mpCacheManager->GpuReset(gpuId);
        if (DCGM_ST_OK != dcgmReturn) 
        {
            dcgmcm_sample_t lwrrentValue;
            
            PRINT_WARNING("%d %d", "For GPU ID %d, reset can't be performed: %d", 
                          gpuId, dcgmReturn);
            
            multiPropertyRetCode++;
            statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_LWRRENT, DCGM_ST_RESET_REQUIRED);
        }
    }

    /* Set Power Limit */
    dcgmReturn = HelperSetPowerLimit(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, dcgmReturn);
    }

    /* Set Perf States */
    dcgmReturn = HelperSetPerfState(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn) 
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, dcgmReturn);
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, dcgmReturn);
    }
    
    /* Set Compute Mode */
    dcgmReturn = HelperSetComputeMode(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn) 
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_COMPUTE_MODE, dcgmReturn);
    }

    /* If any of the operation failed. Return it as an generic error */
    if (0 != multiPropertyRetCode) 
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::EnforceConfigGpu(unsigned int gpuId,
                                                 DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmRet; 
    
    if(!RunningAsRoot())
    {
        PRINT_DEBUG("", "EnforceConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    /* Get the lock for the remainder of this call */
    DcgmLockGuard lockGuard(m_mutex);

    dcgmRet = HelperEnforceConfig(gpuId, statusList);
    if (DCGM_ST_OK != dcgmRet) 
    {
        PRINT_ERROR("%d %d", "Failed to enforce configuration for the GPU Id: %d. Error: %d", gpuId, dcgmRet);
        return dcgmRet;
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperGetSyncBoostInfo(unsigned int *syncBoostIds) 
{
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    DcgmFvBuffer fvBuffer;
    dcgmSyncBoostGroupList_t *pSyncBoostGrpList = 0;
    
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        syncBoostIds[i] = DCGM_INT32_NOT_FOUND;
    }

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_NONE;
    entityPair.entityId = 0; /* Global */
    entities.push_back(entityPair);
    fieldIds.push_back(DCGM_FI_SYNC_BOOST);

    dcgmReturn = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got error %d from GetMultipleLatestLiveSamples()", dcgmReturn);
        return dcgmReturn;
    }

    dcgmBufferedFvLwrsor_t cursor = 0;
    dcgmBufferedFv_t *fv;
    /* There should only be one FV since we have one fieldId above */
    fv = fvBuffer.GetNextFv(&cursor);
    if(!fv)
    {
        PRINT_ERROR("", "Unexpected NULL fv");
        return DCGM_ST_GENERIC_ERROR;
    }

    pSyncBoostGrpList = (dcgmSyncBoostGroupList_t *)fv->value.blob;

    // Assuming a GPU can be part of a single group at point of time
    for (int i = 0; i < pSyncBoostGrpList->numGroups; i++) 
    {
        dcgmSyncBoostGroupListItem_t *pSyncBoostGroupInfo;

        pSyncBoostGroupInfo = &pSyncBoostGrpList->syncBoostGroups[i];

        /* Go through the list of devices and update the syncBoostIds data structures for a 
         * flatten view of groups */
        for (int j = 0; j < pSyncBoostGroupInfo->numDevices; j++) 
        {
            int gpuId = mpCacheManager->LwmlIndexToGpuId(pSyncBoostGroupInfo->lwmlIndex[j]);
            if (DCGM_LWML_ID_BAD == gpuId) 
            {
                PRINT_ERROR("", "Can't get LWML Index from GPU Id");
                continue;
            }

            /* Update RM groupId in the array for bookkeeping */
            syncBoostIds[gpuId] = pSyncBoostGroupInfo->rmGroupId;
        }
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperGPUInSyncBoostGroup(unsigned int gpuIdList[], 
                                                          int count, 
                                                          bool* syncGPUBoostGroup,
                                                          DcgmConfigManagerStatusList *statusList)
{

    dcgmReturn_t dcgmRet;
    
    /* Initialize the syncGpuBoostGroup variable to 0*/
    *syncGPUBoostGroup = 1;
    
    unsigned int syncBoostIds[DCGM_MAX_NUM_DEVICES] = {DCGM_INT32_BLANK};
    
    dcgmRet = HelperGetSyncBoostInfo(syncBoostIds);
    if (DCGM_ST_OK != dcgmRet)
    {
        if(DCGM_ST_NO_DATA == dcgmRet || DCGM_ST_NOT_WATCHED == dcgmRet)
            return DCGM_ST_OK;
        else 
        {
            PRINT_ERROR("","Can't sync Sync Boost Info form cache manager");
            return dcgmRet;
        }
    }

    /* If none of GPUs in the gpuIdList belong to any Sync Boost Group, set syncGPUBoostGroup as 0 and return OK.
       If any of the GPU in gpuIdList doesnt belong to the same Sync Group as others, syncGPUBoostGroup is 1 and return
       DCGM_ST_GPU_IN_SYNC_BOOST_GROUP.
       If all the GPUs are in the same Sync Boost Group, set syncGpuBoostGroup as 1 and return DCGM_ST_OK   
     */
    
    unsigned int rmSyncId = syncBoostIds[gpuIdList[0]];
    bool bGPUsNotInSyncBoostGroup = 1;

    for (int i = 0; i < count; i++) 
    {
        int gpuId = gpuIdList[i];
        /* check if all GPUs belong to the same syncboostGroup or not */
      
        if(!DCGM_INT32_IS_BLANK(syncBoostIds[gpuId]))
        {
            bGPUsNotInSyncBoostGroup = 0;
            if(rmSyncId != syncBoostIds[gpuId])
            {
                PRINT_ERROR("%d %d", "GPU ID %d is part of a different sync group Id %d", gpuId, rmSyncId);
                statusList->AddStatus(gpuId, DCGM_FI_SYNC_BOOST, DCGM_ST_GPU_IN_SYNC_BOOST_GROUP);
                return DCGM_ST_GPU_IN_SYNC_BOOST_GROUP;
            }
        }
    } 
    
    if(bGPUsNotInSyncBoostGroup)
	    *syncGPUBoostGroup = 0;

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetSyncBoost(unsigned int gpuIdList[],
                                                   unsigned int count, 
                                                   unsigned int configFlag,
                                                   DcgmConfigManagerStatusList *statusList)
{
    unsigned int syncBoostId;
    dcgmReturn_t dcgmReturn;

    /* Check if gpus in the gpuIdList are part of any other SyncBoostGroup*/

       
    if ((configFlag != 0) && (configFlag != 1)) {
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }
    

    if (configFlag > 0)
    {
         bool syncGPUBoostGroup = 0;
        /* Check if any of the GPUs in the gpuIdList belongs to any sync boost group*/
    
        dcgmReturn = HelperGPUInSyncBoostGroup(gpuIdList, count, &syncGPUBoostGroup, statusList);
        if(DCGM_ST_OK != dcgmReturn)
        {
            PRINT_ERROR("%d", "Add Sync Boost Failed with %d", dcgmReturn);
            return dcgmReturn;
        }
                
        if( !syncGPUBoostGroup) 
        {
            dcgmReturn = mpCacheManager->AddSyncBoostGrp(gpuIdList, count, &syncBoostId);
            if (DCGM_ST_OK != dcgmReturn) 
            {
                PRINT_ERROR("%d", "Add Sync boost: Failed with %d", dcgmReturn);
                statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, dcgmReturn);
                return dcgmReturn;
            }
        }
    } 
    else 
    {
        unsigned int syncBoostIds[DCGM_MAX_NUM_DEVICES] = {DCGM_INT32_BLANK};
        
        dcgmReturn = HelperGetSyncBoostInfo(syncBoostIds);
        if (DCGM_ST_OK != dcgmReturn)
        {
            PRINT_ERROR("","Remove Sync boost: Can't sync Sync Boost Info form cache manager");
            statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, dcgmReturn);
            return dcgmReturn;
        }
        
        /* 
         * Look for the RM Group Ids for the input GPU id list to check if they all belong to the
         * same RM group ID.
         */
        unsigned int rmSyncIdBase = syncBoostIds[gpuIdList[0]];
        if (DCGM_INT32_IS_BLANK(rmSyncIdBase)) 
        {
            PRINT_ERROR("%d","Remove Sync boost: One of the GPU ID %d is not part of the sync group", 
                        gpuIdList[0]);
            statusList->AddStatus(gpuIdList[0], DCGM_FI_SYNC_BOOST, DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP);
            return DCGM_ST_BADPARAM;
        }
        
        for (unsigned i = 0; i < count; i++) 
        {
            int gpuId = gpuIdList[i];
            
            /* Get the RM ID */
            if (rmSyncIdBase != syncBoostIds[gpuId]) 
            {
                PRINT_ERROR("%d","Remove Sync boost: One of the GPU ID %d is not part of the sync group", gpuId);
                statusList->AddStatus(gpuId, DCGM_FI_SYNC_BOOST, DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP);
                return DCGM_ST_BADPARAM;
            }
        }
        
        dcgmReturn = mpCacheManager->RemoveSyncBoostGrp(rmSyncIdBase);
        if (DCGM_ST_OK != dcgmReturn)
        {
            PRINT_ERROR("%d","Remove Sync boost: Failed to remove sync boost with RM group ID %d", rmSyncIdBase);
            statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, dcgmReturn);
            return dcgmReturn;        
        }
    }
    
    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetSyncBoost(unsigned int gpuIdList[], unsigned int count, 
                                             dcgmConfig_t *setConfig,
                                             DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmReturn;
    
    if(DCGM_INT32_IS_BLANK(setConfig->perfState.syncBoost))
    {
        PRINT_DEBUG("", "syncBoost was blank");
        return DCGM_ST_OK;
    }
    
    if (count <= 1) 
    {
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM);
        PRINT_ERROR("", "Error: At least two GPUs needed to set sync boost");
        return DCGM_ST_BADPARAM;
    }
    
    dcgmReturn = HelperSetSyncBoost(gpuIdList, count, setConfig->perfState.syncBoost,
                                    statusList);
    if (DCGM_ST_OK != dcgmReturn) 
    {
        PRINT_ERROR("%d", "HelperSetSyncBoost returned %d", dcgmReturn);
        return dcgmReturn;
    } 
    else 
    {
        /* Loop through the list of GPUs and update target configuration */
        for (unsigned int i = 0; i < count; i++)
        {
            HelperMergeTargetConfiguration(gpuIdList[i], DCGM_FI_SYNC_BOOST, setConfig);
        }
    }
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::EnforceSyncBoost(unsigned int gpuIdList[], int count, 
                                                 DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t lwcmRet; 
    int i = 0;
    unsigned int targetSyncBoostSetting = 0;
    
    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    /* Check the GPU List to ensure if the target configuration for sync boost is configured
     * for all the GPUs in question
     */
    bool allGpusHaveBlankSyncBoost = true;

    for (i = 0; i < count; i++) 
    {
        unsigned int gpuId = gpuIdList[i];

        if(!m_activeConfig[gpuId])
        {
            PRINT_DEBUG("%d", "Enforce Sync Boost: Target config settings not configured for GPU ID %d", gpuId);
            statusList->AddStatus(gpuId, DCGM_FI_SYNC_BOOST, DCGM_ST_NOT_CONFIGURED);
            return DCGM_ST_OK;
        }
        else if(DCGM_INT32_IS_BLANK(m_activeConfig[gpuId]->perfState.syncBoost))
        {
            PRINT_DEBUG("%d", "Enforce Sync Boost: Blank for GPU ID %d", gpuId);
        }
        else
            allGpusHaveBlankSyncBoost = false;

        if(i == 0)
            targetSyncBoostSetting = m_activeConfig[gpuId]->perfState.syncBoost;
    }

    /* Do we care about sync boost for any GPUs? All blank = nope */
    if(allGpusHaveBlankSyncBoost)
    {
        PRINT_DEBUG("", "All gpus had a blank sync boost");
        return DCGM_ST_OK;
    }

    if (count < 2)
    {
        PRINT_ERROR("", "Error: At least two GPUs needed to set sync boost");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    /**
     * Look at the target sync boost configuration for first GPU in the group (saved above). 
     * If the value is configured, make sure it's consistent for other GPUs in the group.
     */
    
    /**
     * Make sure the target sync boost configuration is consistent for other GPUs in the group before
     * it can be enforced.
     */    
    for (i = 1; i < count; i++) 
    {
        unsigned int syncBoostSetting;
        unsigned int gpuId = gpuIdList[i];

        syncBoostSetting = m_activeConfig[gpuId]->perfState.syncBoost;

        if (targetSyncBoostSetting != syncBoostSetting) 
        {
            statusList->AddStatus(gpuId, DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM);
            PRINT_ERROR("%d", "Enforce Sync Boost: Target Sync boost settings not consistent across the group."
                        "First outlier GPU ID: %d", gpuId);
            return DCGM_ST_BADPARAM;
        }
    }

    /* Check if gpus in the gpuIdList are part of any other SyncBoostGroup*/
    bool syncGPUBoostGroup = 0;
      
    lwcmRet = HelperGPUInSyncBoostGroup(gpuIdList, count, &syncGPUBoostGroup, statusList);
    
    if(DCGM_ST_OK != lwcmRet && DCGM_ST_GPU_IN_SYNC_BOOST_GROUP != lwcmRet)
    {
        PRINT_ERROR("%d", "Enforce Sync Boost Failed with %d", lwcmRet);
        return lwcmRet;
    }

    /* If set sync is enforced, check if GPUs in the gpuIdList are a part of Sync GPU Boost Group */
    if(targetSyncBoostSetting)
    {
        /* If all the GPUs in the gpuIdList belong to the same Sync boost group, return SUCCESS */
        if(DCGM_ST_OK == lwcmRet && syncGPUBoostGroup) 
        {
            PRINT_INFO("", "All the GPUs are part of the same sync boost group");
            return DCGM_ST_OK;
        }
        else
        {
            PRINT_DEBUG("", "One or more gpus in the the gpuIDList are part of different Sync Boost Groups");
            return DCGM_ST_OK;
        }
    }
    else
    {
        /* If remove sync group is enforced, check if a Sync GPU Boost Group exist */
        if(DCGM_ST_OK == lwcmRet && !syncGPUBoostGroup)
        {
            /* If none of the GPUs in the gpuIdList belong to any sync boost group, return SUCCESS */
            PRINT_INFO("", "No Sync Boost Group to remove ");
            return DCGM_ST_OK;
        }
        else
        {
            PRINT_DEBUG("", "One or more gpus in the the gpuIDList are part of different Sync Boost Groups");
            return DCGM_ST_OK;
        }
    }
        
    /**
     * At this point, it can be said that the target sync configuration is consistent for the 
     * group. Ilwoke Set API to enforce sync boost settings
     */
    lwcmRet = HelperSetSyncBoost(gpuIdList, count, targetSyncBoostSetting, statusList);
    return lwcmRet;
}

/*****************************************************************************/
void DcgmConfigManager::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetConfig(unsigned int groupId, dcgmConfig_t *setConfig, 
                                          DcgmConfigManagerStatusList *statusList)

{
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int>gpuIds;
    unsigned int index;

    /* GroupId was already validated by the caller */
    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_DEBUG("%d %u", "GetGroupEntities returned %d for groupId %u",
                    dcgmReturn, groupId);
        return dcgmReturn;
    }

    /* Config manager only works on globals and GPUs. The sync boost call needs all GPU IDs in a contiguous array,
       so aggregate them in gpuIds[] */
    for (index = 0; index < entities.size(); index++) 
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[index].entityId);
    }

    /* Get number of gpus from thr group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        PRINT_ERROR("%d", "Config Set Err: No gpus configured for the group id : %d", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    int grpRetCode = 0;
    
    /* Check if group level power budget is specified */
    if (!DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
    {
        if (setConfig->powerLimit.type == DCGM_CONFIG_POWER_BUDGET_GROUP)
        {
            setConfig->powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
            setConfig->powerLimit.val /= gpuIds.size();
            PRINT_DEBUG("%d %u", "Divided our group power limit by %d. is now %u",
                        (int)gpuIds.size(), setConfig->powerLimit.val);
        }
    }

    /* Loop through the group to set configuration for each GPU */
    for (index = 0; index < gpuIds.size(); index++) 
    {   
        unsigned int gpuId = gpuIds[index];
        dcgmReturn = SetConfigGpu(gpuId, setConfig, statusList);
        if (DCGM_ST_OK != dcgmReturn)
        {
            PRINT_ERROR("%d %u", "SetConfig failed with %d for gpuId %u", dcgmReturn, gpuId);
            grpRetCode++;
        }
    }

    /* Special handling for sync boost */
    dcgmReturn = SetSyncBoost(&gpuIds[0], gpuIds.size(), setConfig, statusList);
    if (DCGM_ST_OK != dcgmReturn)
        grpRetCode++;

    if (grpRetCode)
        return DCGM_ST_GENERIC_ERROR;
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::EnforceConfigGroup(unsigned int groupId, 
                                                   DcgmConfigManagerStatusList *statusList)

{
    int index;
    unsigned int grpRetCode = 0;
    std::vector<dcgmGroupEntityPair_t>entities;
    std::vector<unsigned int>gpuIds;
    dcgmReturn_t dcgmReturn;

    /* The caller already verified and updated the groupId */

    dcgmReturn = mpGroupManager->GetGroupEntities(0, groupId, entities);
    if(dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupGpuIds()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Config manager only works on globals and GPUs. The sync boost call needs all GPU IDs in a contiguous array,
       so aggregate them in gpuIds[] */
    for (index = 0; index < (int)entities.size(); index++) 
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;
        
        gpuIds.push_back(entities[index].entityId);
    }

    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        PRINT_ERROR("%d", "Config Enforce Err: No GPUs configured for the group id : %d", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    /* Loop through the group to set configuration for each GPU */
    for (index = 0; index < (int)gpuIds.size(); index++)
    {
        unsigned int gpuId;
        gpuId = gpuIds[index];
        dcgmReturn = EnforceConfigGpu(gpuId, statusList);
        if (DCGM_ST_OK != dcgmReturn)
            grpRetCode++;
    }

    /* Special handling for sync boost */
    dcgmReturn = EnforceSyncBoost(&gpuIds[0], gpuIds.size(), statusList);
    if (DCGM_ST_OK != dcgmReturn)
        grpRetCode++;

    if (0 == grpRetCode)
        return DCGM_ST_OK;
    else
        return DCGM_ST_GENERIC_ERROR;
}

/*****************************************************************************/

