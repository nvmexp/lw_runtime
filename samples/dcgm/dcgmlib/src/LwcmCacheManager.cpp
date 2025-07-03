
#define LWML_INIT_UUID
#include "LwcmCacheManager.h"
#include "lwml_internal.h"
#include "lwos.h"
#include "logging.h"
#include "lwml.h"
#include "lwml_grid.h"
#include <string>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <stdexcept> //std::runtime_error
#include <stdint.h>
#include <math.h>
#include "DcgmMutex.h"
#include "MurmurHash3.h"

/*****************************************************************************/
/* Conditional / Debug Features */
//#define DEBUG_UPDATE_LOOP 1

/*****************************************************************************/
/* Keyed vector definition for pids that we've seen */
typedef struct dcgmcm_pid_seen_t
{
    unsigned int pid;
    timelib64_t timestamp;
} dcgmcm_pid_seen_t, *dcgmcm_pid_seen_p;

static int dcgmcm_pidSeenCmpCB(void *L, void *R)
{
    dcgmcm_pid_seen_p l = (dcgmcm_pid_seen_p)L;
    dcgmcm_pid_seen_p r = (dcgmcm_pid_seen_p)R;

    if(l->pid < r->pid)
        return -1;
    else if(l->pid > r->pid)
        return 1;

    if(l->timestamp < r->timestamp)
        return -1;
    else if(l->timestamp > r->timestamp)
        return 1;

    return 0;
}

static int dcgmcm_pidSeenMergeCB(void *L, void *R)
{
    dcgmcm_pid_seen_p l = (dcgmcm_pid_seen_p)L;
    dcgmcm_pid_seen_p r = (dcgmcm_pid_seen_p)R;

    if(l->pid < r->pid)
        return -1;
    else if(l->pid > r->pid)
        return 1;

    if(l->timestamp < r->timestamp)
        return -1;
    else if(l->timestamp > r->timestamp)
        return 1;

    return 0;
}

static int dcgmcm_pidSeenMergeCB(void *current, void *inserting, void *user)
{
    PRINT_ERROR("", "Unexpected dcgmcm_pidSeenMergeCB");
    return KV_ST_DUPLICATE;
}

/*****************************************************************************/
/* Hash callbacks for m_entityWatchHashTable */
static unsigned int entityKeyHashCB(const void *key)
{
    unsigned int retVal = 0;

    /* We're passing address to key because the pointer passed in is the actual value */
    MurmurHash3_x86_32(&key, sizeof(key), 0, &retVal);
    return retVal;
}

/* Comparison callbacks for m_entityWatchHashTable */
static int entityKeyCmpCB(const void *key1, const void *key2)
{
    if (key1 == key2)
        return 1; /* Yes. The caller expects 1 for == */
    else
        return 0;
}

static void entityValueFreeCB(void *WatchInfo)
{
    dcgmcm_watch_info_p watchInfo = (dcgmcm_watch_info_p)WatchInfo;

    if(!watchInfo)
    {
        PRINT_ERROR("", "FreeWatchInfo got NULL watchInfo");
        return;
    }

    if(watchInfo->timeSeries)
    {
        timeseries_destroy(watchInfo->timeSeries);
        watchInfo->timeSeries = 0;
    }

    delete(watchInfo);
}

/*****************************************************************************/
// NOTE: LWML is initialized by LwcmHostEngineHandler before DcgmCacheManager is instantiated
DcgmCacheManager::DcgmCacheManager() : m_pollInLockStep(0), m_maxSampleAgeUsec((timelib64_t)3600 * 1000000),
                                       m_numGpus(0), m_lwmlInitted(true), m_inDriverCount(0),
                                       m_waitForDriverClearCount(0), m_lwmlEventSetInitialized(false)
{
    int i, kvSt = 0;

    m_entityWatchHashTable = 0;
    m_haveAnyLiveSubscribers = false;
    
    m_mutex = new DcgmMutex(0);
    //m_mutex->EnableDebugLogging(true);

    lwosCondCreate(&m_startUpdateCondition);
    lwosCondCreate(&m_updateCompleteCondition);

    memset(&m_runStats, 0, sizeof(m_runStats));

    m_entityWatchHashTable = hashtable_create(entityKeyHashCB, entityKeyCmpCB, 0, 
                                              entityValueFreeCB);
    if(!m_entityWatchHashTable)
    {
        PRINT_CRITICAL("", "hashtable_create failed");
        throw std::runtime_error("DcgmCacheManager failed to create its hashtable.");
    }

    m_etblLwmlCommonInternal = 0;

    memset(&m_lwrrentEventMask[0], 0, sizeof(m_lwrrentEventMask));

    memset(m_gpus, 0, sizeof(m_gpus));

    m_numLwSwitches = 0;
    memset(m_lwSwitches, 0, sizeof(m_lwSwitches));

    m_accountingPidsSeen = keyedvector_alloc(sizeof(dcgmcm_pid_seen_t), 0,
                                             dcgmcm_pidSeenCmpCB,
                                             dcgmcm_pidSeenMergeCB, 0,
                                             0, &kvSt);

    for (unsigned short fieldId = 1; fieldId < DCGM_FI_MAX_FIELDS; ++fieldId)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
        if (fieldMeta && fieldMeta->fieldId != DCGM_FI_UNKNOWN)
        {
            m_allValidFieldIds.push_back(fieldId);
        }
    }
    
    m_eventThread = new DcgmCacheManagerEventThread(this);
}

/*****************************************************************************/
DcgmCacheManager::~DcgmCacheManager()
{
    Shutdown("");
    
    if(m_mutex)
        delete(m_mutex);
    m_mutex = 0;

    lwosCondDestroy(&m_startUpdateCondition);
    lwosCondDestroy(&m_updateCompleteCondition);

    if(m_entityWatchHashTable)
    {
        hashtable_destroy(m_entityWatchHashTable);
        m_entityWatchHashTable = 0;
    }

    if(m_accountingPidsSeen)
    {
        keyedvector_destroy(m_accountingPidsSeen);
        m_accountingPidsSeen = 0;
    }

    UninitializeLwmlEventSet();
}

/*****************************************************************************/
void DcgmCacheManager::UninitializeLwmlEventSet()
{

    if (m_lwmlEventSetInitialized)
    {
        lwmlEventSetFree(m_lwmlEventSet);
        m_lwmlEventSetInitialized = false;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::InitializeLwmlEventSet()
{
    if (!m_lwmlEventSetInitialized)
    {
        lwmlReturn_t lwmlReturn = lwmlEventSetCreate(&m_lwmlEventSet);
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%s", "Error %s from lwmlEventSetCreate", lwmlErrorString(lwmlReturn));
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }
        m_lwmlEventSetInitialized = true;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::GetGlobalWatchInfo(unsigned int fieldId,
                                                         int createIfNotExists)
{
    return GetEntityWatchInfo(DCGM_FE_NONE, 0, fieldId, createIfNotExists);
}

/*****************************************************************************/
int DcgmCacheManager::IsGpuWhitelisted(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpuInfo;
    static int haveReadElw = 0;
    static int bypassWhitelist = 0;
    lwmlChipArchitecture_t minChipArch;

    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u", "Invalid gpuId %u to IsGpuWhitelisted", gpuId);
        return 0;
    }

    /* First, see if we're bypassing the whitelist */
    if(!haveReadElw)
    {
        haveReadElw = 1;
        if(-1 < lwosGetElw(DCGM_ELW_WL_BYPASS, 0, 0))
        {
            PRINT_DEBUG("", "Whitelist bypassed with elw variable");
            bypassWhitelist = 1;
        }
        else
        {
            PRINT_DEBUG("", "Whitelist NOT bypassed with elw variable");
            bypassWhitelist = 0;
        }
    }

    if(bypassWhitelist)
    {
        PRINT_DEBUG("%u", "gpuId %u whitelisted due to elw bypass", gpuId);
        return 1;
    }

    gpuInfo = &m_gpus[gpuId];

    /* Check our chip architecture against DCGM's minimum supported arch.
       This is Kepler for Tesla GPUs and Maxwell for everything else */
    minChipArch = LWML_CHIP_ARCH_MAXWELL;
    if(gpuInfo->brand == DCGM_GPU_BRAND_TESLA)
    {
        PRINT_DEBUG("%u", "gpuId %u is a Tesla GPU", gpuId);
        minChipArch = LWML_CHIP_ARCH_KEPLER;
    }

    if(gpuInfo->arch >= minChipArch)
    {
        PRINT_DEBUG("%u %u", "gpuId %u, arch %u is whitelisted.", gpuId, gpuInfo->arch);
        return 1;
    }
    else
    {
        PRINT_DEBUG("%u %u", "gpuId %u, arch %u is NOT whitelisted.", gpuId, gpuInfo->arch);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateLwLinkLinkState(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpu;
    lwmlReturn_t lwmlSt;
    unsigned int linkId;
    lwmlEnableState_t isActive = LWML_FEATURE_DISABLED;

    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    gpu = &m_gpus[gpuId];

    if(gpu->status == DcgmcmGpuStatusFakeGpu)
    {
        PRINT_DEBUG("%u", "Skipping UpdateLwLinkLinkState for fake gpuId %u", gpuId);
        return DCGM_ST_OK;
    }

    for(linkId = 0; linkId < DCGM_LWLINK_MAX_LINKS_PER_GPU; linkId++)
    {
        lwmlSt = lwmlDeviceGetLwLinkState(gpu->lwmlDevice, linkId, &isActive);
        if(lwmlSt == LWML_ERROR_NOT_SUPPORTED)
        {
            PRINT_DEBUG("%u %u", "gpuId %u, LwLink laneId %u Not supported.", gpuId, linkId);
            gpu->lwLinkLinkState[linkId] = DcgmLwLinkLinkStateNotSupported;
            continue;
        }
        else if(lwmlSt != LWML_SUCCESS)
        {
            PRINT_DEBUG("%u %u %d", "gpuId %u, LwLink laneId %u. lwmlSt: %d.", gpuId, linkId, (int)lwmlSt);
            /* Treat any error as NotSupported. This is important for Volta vs Pascal where lanes 5+6 will
             * work for Volta but will return invalid parameter for Pascal
             */
            gpu->lwLinkLinkState[linkId] = DcgmLwLinkLinkStateNotSupported;
            continue;
        }

        if (isActive == LWML_FEATURE_DISABLED)
        {
            PRINT_DEBUG("%u %u", "gpuId %u, LwLink laneId %u. DOWN", gpuId, linkId);
            gpu->lwLinkLinkState[linkId] = DcgmLwLinkLinkStateDown;
        }
        else
        {
            PRINT_DEBUG("%u %u", "gpuId %u, LwLink laneId %u. UP", gpuId, linkId);
            gpu->lwLinkLinkState[linkId] = DcgmLwLinkLinkStateUp;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuBlacklist(std::vector<lwmlBlacklistDeviceInfo_t> &blacklist)
{
    blacklist = m_gpuBlacklist;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ReadAndCacheGpuBlacklist(void)
{
    lwmlReturn_t lwmlSt;
    unsigned int i, blacklistCount = 0;
    lwmlBlacklistDeviceInfo_t blacklistEntry;

    lwmlSt = lwmlGetBlacklistDeviceCount(&blacklistCount);
    if(lwmlSt == LWML_ERROR_FUNCTION_NOT_FOUND)
    {
        PRINT_INFO("", "lwmlGetBlacklistDeviceCount(). was not found. Driver is likely older than r400.");
        return DCGM_ST_NOT_SUPPORTED;
    }
    else if(lwmlSt != LWML_SUCCESS)
    {
        PRINT_ERROR("%d", "lwmlGetBlacklistDeviceCount returned %d", (int)lwmlSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    PRINT_INFO("%u", "Got %u blacklisted GPUs", blacklistCount);

    /* Start over since we're reading the blacklist again */
    m_gpuBlacklist.clear();

    for(i = 0; i < blacklistCount; i++)
    {
        memset(&blacklistEntry, 0, sizeof(blacklistEntry));

        lwmlSt = lwmlGetBlacklistDeviceInfoByIndex(i, &blacklistEntry);
        if(lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%u %d", "lwmlGetBlacklistDeviceInfoByIndex(%u) returned %d", 
                        i, (int)lwmlSt);
            continue;
        }

        PRINT_INFO("%s %s", "Read GPU blacklist entry PCI %s, UUID %s", 
                   blacklistEntry.pciInfo.busId, blacklistEntry.uuid);

        m_gpuBlacklist.push_back(blacklistEntry);
    }

    return DCGM_ST_OK;
}
    
/*****************************************************************************/
void DcgmCacheManager::SynchronizeDriverEntries(unsigned int &countToWaitFor, unsigned int &queuedCount,
                                                bool entering)
{
    bool waited = false;

    // Spin until all threads leave the driver, and then exit
    // If we are entering, wait until our LWML event set is initialized as well
    while (countToWaitFor > 0 && (!entering || m_lwmlEventSetInitialized))
    {
        waited = true;
        queuedCount++;
        dcgm_mutex_unlock(m_mutex);
        usleep(100);
        dcgm_mutex_lock(m_mutex);
    }
    
    // We can decrement this since this thread is no longer waiting and will keep
    // m_mutex locked throughout, but only decrement if we entered the loop
    if (waited)
        queuedCount--;
}

/*****************************************************************************/
void DcgmCacheManager::WaitForThreadsToExitDriver()
{
    SynchronizeDriverEntries(m_inDriverCount, m_waitForDriverClearCount, false);
}

/*****************************************************************************/
void DcgmCacheManager::WaitForDriverToBeReady()
{
    SynchronizeDriverEntries(m_waitForDriverClearCount, m_inDriverCount, true);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::DetachGpus()
{
    // Make an empty vgpuInstanceCount for clearing the vgpu list.
    lwmlVgpuInstance_t vgpuInstanceCount = 0;
    dcgm_mutex_lock(m_mutex);

    WaitForThreadsToExitDriver();

    if (m_lwmlInitted == false)
    {
        // already uninitialized
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_OK;
    }

    UninitializeLwmlEventSet();

    lwmlReturn_t lwmlSt = lwmlShutdown();
    if (lwmlSt != LWML_SUCCESS)
    {
        if (lwmlSt == LWML_ERROR_UNINITIALIZED)
            m_lwmlInitted = false;

        PRINT_ERROR("%d", "lwmlShutdown returned %d", (int)lwmlSt);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }
    m_lwmlInitted = false;

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        m_gpus[i].status = DcgmcmGpuStatusDetached; // Should we use an existing status?
    }

    dcgm_mutex_unlock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
        ManageVgpuList(m_gpus[i].gpuId, &vgpuInstanceCount);
        
    return DCGM_ST_OK;
}

/*****************************************************************************/
// NOTE: This method assumes that m_mutex is locked as it's going to manipulate m_gpus
void DcgmCacheManager::MergeNewlyDetectedGpuList(dcgmcm_gpu_info_p detectedGpus, int count)
{
    if (m_numGpus == 0)
    {
        // This is the first time we're running this function, just copy the results completely.
        memcpy(m_gpus, detectedGpus, sizeof(dcgmcm_gpu_info_t) * count);
        m_numGpus = count;
    }
    else
    {
        // Match each detected GPU to existing GPUs by uuid
        std::vector<int> unmatchedIndices;

        // Update the list from the GPUs that are lwrrently detected
        for (int detectedIndex = 0; detectedIndex < count; detectedIndex++)
        {
            bool matched = false;

            for (unsigned int existingIndex = 0; existingIndex < m_numGpus; existingIndex++)
            {
                if (!strcmp(detectedGpus[detectedIndex].uuid, m_gpus[existingIndex].uuid))
                {
                    m_gpus[existingIndex].lwmlIndex = detectedGpus[detectedIndex].lwmlIndex;
                    m_gpus[existingIndex].lwmlDevice = detectedGpus[detectedIndex].lwmlDevice;
                    m_gpus[existingIndex].brand = detectedGpus[detectedIndex].brand;
                    m_gpus[existingIndex].arch = detectedGpus[detectedIndex].arch;
                    memcpy(&m_gpus[existingIndex].pciInfo, &detectedGpus[detectedIndex], sizeof(lwmlPciInfo_t));

                    // Found a match, turn this GPU back on
                    m_gpus[existingIndex].status = DcgmcmGpuStatusOk;
                    matched = true;
                    break;
                }
            }

            if (matched == false)
                unmatchedIndices.push_back(detectedIndex);
        }

        // Add in new GPUs that weren't previously detected
        for (size_t i = 0; i < unmatchedIndices.size() && m_numGpus < DCGM_MAX_NUM_DEVICES; i++)
        {
            // Copy each new GPU after the ones that are previously detected
            memcpy(m_gpus + m_numGpus, detectedGpus + unmatchedIndices[i],
                    sizeof(dcgmcm_gpu_info_t));

            // Make sure we have unique gpuIds
            m_gpus[m_numGpus].gpuId = m_numGpus;

            m_numGpus++;
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AttachGpus()
{
    lwmlReturn_t lwmlSt;
    unsigned int lwmlDeviceCount = 0;
    int               detectedGpusCount = 0;
    dcgmcm_gpu_info_t detectedGpus[DCGM_MAX_NUM_DEVICES]; /* All of the GPUs we know about, indexed by gpuId */
    memset(&detectedGpus, 0, sizeof(detectedGpus));
    int               oldNumGpus = 0;
    dcgmReturn_t      ret;

    dcgm_mutex_lock(m_mutex);

    // Generally speaking this will be true every time except the first time this is called
    if (m_lwmlInitted == false)
    {
        lwmlSt = lwmlInit();
        if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d", "lwmlInit returned %d", (int)lwmlSt);
            dcgm_mutex_unlock(m_mutex);
            return DCGM_ST_GENERIC_ERROR;
        }

        m_lwmlInitted = true;
    }

    lwmlSt = lwmlDeviceGetCount(&lwmlDeviceCount);
    if (lwmlSt != LWML_SUCCESS)
    {
        PRINT_ERROR("%d", "lwmlDeviceGetCount returned %d", (int)lwmlSt);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    if (lwmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%u %d", "More LWML devices (%u) than DCGM_MAX_NUM_DEVICES (%d)",
                    lwmlDeviceCount, DCGM_MAX_NUM_DEVICES);
        /* Keep going. Just fill up to our limit */
    }
    detectedGpusCount = DCGM_MIN(lwmlDeviceCount, DCGM_MAX_NUM_DEVICES);

    ret = InitializeLwmlEventSet();
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Couldn't create the proper LWML event set when re-attaching to GPUS: %s",
                    errorString(ret));
    }

    for (int i = 0; i < detectedGpusCount; i++)
    {
        detectedGpus[i].gpuId = i; /* For now, gpuId == index == lwmlIndex */
        detectedGpus[i].lwmlIndex = i;
        detectedGpus[i].status = DcgmcmGpuStatusOk; /* Start out OK */

        lwmlSt = lwmlDeviceGetHandleByIndex(detectedGpus[i].lwmlIndex, &detectedGpus[i].lwmlDevice);

        // if lwmlReturn == LWML_ERROR_NO_PERMISSION this is ok
        // but it should be logged in case it is unexpected
        if (lwmlSt == LWML_ERROR_NO_PERMISSION)
        {
            PRINT_WARNING("%d", "GPU %d initialization was skipped due to no permissions.", i);
            detectedGpus[i].status = DcgmcmGpuStatusInaccessible;
            continue;
        }
        else if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetHandleByIndex of lwmlIndex %d",
                        (int)lwmlSt, i);
            /* Treat this error as inaccessible */
            detectedGpus[i].status = DcgmcmGpuStatusInaccessible;
            continue;
        }

        lwmlSt = lwmlDeviceGetUUID(detectedGpus[i].lwmlDevice, detectedGpus[i].uuid, sizeof(detectedGpus[i].uuid));
        if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetUUID of lwmlIndex %d",
                        (int)lwmlSt, i);
            /* Non-fatal. Keep going. */
        }

        lwmlBrandType_t lwmlBrand = LWML_BRAND_UNKNOWN;
        lwmlSt = lwmlDeviceGetBrand(detectedGpus[i].lwmlDevice, &lwmlBrand);
        if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetBrand of lwmlIndex %d",
                        (int)lwmlSt, i);
            /* Non-fatal. Keep going. */
        }
        detectedGpus[i].brand = (dcgmGpuBrandType_t)lwmlBrand;

        lwmlSt = lwmlDeviceGetPciInfo(detectedGpus[i].lwmlDevice, &detectedGpus[i].pciInfo);
        if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetPciInfo of lwmlIndex %d",
                        (int)lwmlSt, i);
            /* Non-fatal. Keep going. */
        }

        /* Read the arch before we check the whitelist since the arch is used for the whitelist */
        lwmlSt = LWML_CALL_ETBL(m_etblLwmlCommonInternal, DeviceGetChipArchitecture,
            (detectedGpus[i].lwmlDevice, &detectedGpus[i].arch));
        if (lwmlSt != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got lwml error %d from lwmlDeviceGetChipArchitecture of lwmlIndex %d",
                        (int)lwmlSt, i);
            /* Non-fatal. Keep going. */
        }
    }

    MergeNewlyDetectedGpuList(detectedGpus, detectedGpusCount);

    /* We keep track of all GPUs that LWML knew about.
     * Do this before the for loop so that IsGpuWhitelisted doesn't
     * think we are setting invalid gpuIds */

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmcmGpuStatusDetached || m_gpus[i].status == DcgmcmGpuStatusInaccessible)
            continue;

        if (!IsGpuWhitelisted(m_gpus[i].gpuId))
        {
            PRINT_DEBUG("%u", "gpuId %u is NOT whitelisted.", m_gpus[i].gpuId);
            m_gpus[i].status = DcgmcmGpuStatusUnsupported;
        }

        UpdateLwLinkLinkState(m_gpus[i].gpuId);
    }

    /* Read and cache the GPU blacklist on each attach */
    ReadAndCacheGpuBlacklist();
    
    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::AddFakeGpu(unsigned int pciDeviceId, unsigned int pciSubSystemId)
{
    unsigned int gpuId = DCGM_GPU_ID_BAD;
    dcgmcm_gpu_info_t *gpuInfo;
    int i;
    dcgmReturn_t dcgmReturn;
    dcgmcm_sample_t sample;

    if (m_numGpus >= DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%d", "Could not add another GPU. Already at limit of %d", DCGM_MAX_NUM_DEVICES);
        return gpuId; /* Too many already */
    }

    dcgm_mutex_lock(m_mutex);

    gpuId = m_numGpus;
    gpuInfo = &m_gpus[gpuId];

    gpuInfo->brand = DCGM_GPU_BRAND_TESLA;
    gpuInfo->gpuId = gpuId;
    gpuInfo->lwmlIndex = gpuId;
    memset(&gpuInfo->pciInfo, 0, sizeof(gpuInfo->pciInfo));
    gpuInfo->pciInfo.pciDeviceId = pciDeviceId;
    gpuInfo->pciInfo.pciSubSystemId = pciSubSystemId;
    gpuInfo->status = DcgmcmGpuStatusFakeGpu;
    memset(gpuInfo->uuid, 0, sizeof(gpuInfo->uuid));
    for(i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_GPU; i++)
    {
        gpuInfo->lwLinkLinkState[i] = DcgmLwLinkLinkStateNotSupported;
    }

    m_numGpus++;
    dcgm_mutex_unlock(m_mutex);

    /* Inject ECC mode as enabled so policy management works */
    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();
    sample.val.i64 = 1;

    dcgmReturn = InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_LWRRENT, &sample, 1);
    if(dcgmReturn != DCGM_ST_OK)
        PRINT_ERROR("%d", "Error %d from InjectSamples()", (int)dcgmReturn);

    return gpuId;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::AddFakeGpu(void)
{
    return AddFakeGpu(0, 0);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo)
{
    unsigned int i;

    /* Acquire the lock for consistency */
    DcgmLockGuard dlg(m_mutex);

    gpuInfo.resize(m_numGpus);

    for(i = 0; i < m_numGpus; i++)
    {
        gpuInfo[i].gpuId = m_gpus[i].gpuId;
        gpuInfo[i].status = m_gpus[i].status;
        gpuInfo[i].brand = m_gpus[i].brand;
        gpuInfo[i].lwmlIndex = m_gpus[i].lwmlIndex;
        gpuInfo[i].pciInfo = m_gpus[i].pciInfo;
        gpuInfo[i].arch = m_gpus[i].arch;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmCacheManager::GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, 
                                          dcgm_field_eid_t entityId)
{
    int i;

    switch(entityGroupId)
    {
        case DCGM_FE_GPU:
        {
            return (entityId >= 0 && entityId < m_numGpus);
        }
        
        case DCGM_FE_SWITCH:
        {
            for(int i = 0; i < m_numLwSwitches; i++)
            {
                if(entityId == m_lwSwitches[i].physicalId)
                    return true;
            }
            return false;
        }
        
        case DCGM_FE_VGPU: //Intentional fall-through
        default:
            PRINT_DEBUG("%u %u", "GetIsValidEntityId not supported for entityGroup %u, entityId %u", 
                        entityGroupId, entityId);
            return false;
    }
}

/*****************************************************************************/
unsigned int DcgmCacheManager::AddFakeLwSwitch(void)
{
    dcgmcm_lwswitch_info_t *lwSwitch = NULL;
    unsigned int entityId = DCGM_ENTITY_ID_BAD;
    int i;
    
    dcgm_mutex_lock(m_mutex);

    if(m_numLwSwitches >= DCGM_MAX_NUM_SWITCHES)
    {
        PRINT_ERROR("%d", "Could not add another LwSwitch. Already at limit of %d", 
                    DCGM_MAX_NUM_SWITCHES);
        dcgm_mutex_unlock(m_mutex);
        return entityId; /* Too many already */
    }

    lwSwitch = &m_lwSwitches[m_numLwSwitches];

    lwSwitch->status = DcgmcmGpuStatusFakeGpu;

    /* Assign a physical ID based on trying to find one that isn't in use yet */
    for(lwSwitch->physicalId = 0; 
        lwSwitch->physicalId < DCGM_ENTITY_ID_BAD; 
        lwSwitch->physicalId++)
    {
        if(!GetIsValidEntityId(DCGM_FE_SWITCH, lwSwitch->physicalId))
            break;
    }

    PRINT_DEBUG("%u", "AddFakeLwSwitch allocating physicalId %u", lwSwitch->physicalId);
    entityId = lwSwitch->physicalId;

    /* Set the link state to Disconnected rather than Unsupported since LwSwitches support LwLink */
    for(i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; i++)
    {
        lwSwitch->lwLinkLinkState[i] = DcgmLwLinkLinkStateDisabled;
    }

    m_numLwSwitches++;

    dcgm_mutex_unlock(m_mutex);
    return entityId; 
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddLwSwitch(unsigned int physicalId)
{
    dcgmcm_lwswitch_info_t *lwSwitch = NULL;
    int i;
    
    dcgm_mutex_lock(m_mutex);

    /* First, see if this LwSwitch already exists in our table. 
       Same physicalId == same LwSwitch */
    for(i = 0; i < m_numLwSwitches; i++)
    {
        lwSwitch = &m_lwSwitches[i];
        if(lwSwitch->physicalId == physicalId)
        {
            /* We already have it. No problem */
            dcgm_mutex_unlock(m_mutex);
            PRINT_DEBUG("%u %d", "LwSwitch physicalId %u is already present at index %d",
                        physicalId, i);
            return DCGM_ST_OK;
        }
    }

    if(m_numLwSwitches >= DCGM_MAX_NUM_SWITCHES)
    {
        PRINT_ERROR("%d", "Could not add another LwSwitch. Already at limit of %d", 
                    DCGM_MAX_NUM_SWITCHES);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR; /* Too many already */
    }

    lwSwitch = &m_lwSwitches[m_numLwSwitches];

    lwSwitch->physicalId = physicalId;
    lwSwitch->status = DcgmcmGpuStatusOk;

    /* Set the link state to Disconnected rather than Unsupported since LwSwitches support LwLink */
    for(i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; i++)
    {
        lwSwitch->lwLinkLinkState[i] = DcgmLwLinkLinkStateDisabled;
    }

    PRINT_DEBUG("%u %d", "AddLwSwitch added physicalId %u at index %d", 
                lwSwitch->physicalId, m_numLwSwitches);
    m_numLwSwitches++;

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK; 
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetLwSwitchLinkState(unsigned int physicalId, 
                                                    unsigned int portIndex,
                                                    dcgmLwLinkLinkState_t linkState)
{
    dcgmcm_lwswitch_info_t *lwSwitch = NULL;
    int i;

    if(portIndex >= DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH)
    {
        PRINT_ERROR("%u", "SetLwSwitchLinkState called for invalid portIndex %u", portIndex);
        return DCGM_ST_BADPARAM;
    }

    /* Is the physical ID valid? */
    for(i = 0; i < m_numLwSwitches; i++)
    {
        if(m_lwSwitches[i].physicalId == physicalId)
        {
            /* Found it */
            lwSwitch = &m_lwSwitches[i];
            break;
        }
    }

    if(!lwSwitch)
    {
        PRINT_ERROR("%u", "SetLwSwitchLinkState called for invalid physicalId %u", physicalId);
        return DCGM_ST_BADPARAM;
    }

    PRINT_INFO("%u %u %u", "Setting LwSwitch physicalId %u, port %u to link state %u", 
               physicalId, portIndex, linkState);
    lwSwitch->lwLinkLinkState[portIndex] = linkState;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetGpuLwLinkLinkState(unsigned int gpuId, 
                                                     unsigned int linkId,
                                                     dcgmLwLinkLinkState_t linkState)
{
    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u", "Bad gpuId %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    if(linkId >= DCGM_LWLINK_MAX_LINKS_PER_GPU)
    {
        PRINT_ERROR("%u", "SetGpuLwLinkLinkState called for invalid linkId %u", linkId);
        return DCGM_ST_BADPARAM;
    }

    PRINT_INFO("%u %u %u", "Setting gpuId %u, link %u to link state %u", 
               gpuId, linkId, linkState);
    m_gpus[gpuId].lwLinkLinkState[linkId] = linkState;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetEntityLwLinkLinkState(dcgm_field_entity_group_t entityGroupId,
                                                        dcgm_field_eid_t entityId, 
                                                        unsigned int linkId,
                                                        dcgmLwLinkLinkState_t linkState)
{
    

    if(entityGroupId == DCGM_FE_GPU)
        return SetGpuLwLinkLinkState(entityId, linkId, linkState);
    else if(entityGroupId == DCGM_FE_SWITCH)
        return SetLwSwitchLinkState(entityId, linkId, linkState);
    else
    {
        PRINT_ERROR("%u", "entityGroupId %u does not support setting LwLink link state",
                    entityGroupId);
        return DCGM_ST_NOT_SUPPORTED;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Init(int pollInLockStep, double maxSampleAge)
{
    lwmlReturn_t lwmlSt;

    m_pollInLockStep = pollInLockStep;

    lwmlSt = lwmlInternalGetExportTable((const void**)&m_etblLwmlCommonInternal,
                                        &ETID_LWMLCommonInternal);
    if(lwmlSt != LWML_SUCCESS)
    {
        PRINT_ERROR("%s", "Unable to get callback table. lwml error: %s",
                    lwmlErrorString(lwmlSt));
        return DCGM_ST_INIT_ERROR;
    }

    AttachGpus();

    /* Start the event watch before we start the event reading thread */
    ManageDeviceEvents(DCGM_GPU_ID_BAD, 0);

    if(!m_eventThread)
    {
        PRINT_ERROR("", "m_eventThread was NULL. We're unlikely to collect any events.");
        return DCGM_ST_GENERIC_ERROR;
    }
    int st = m_eventThread->Start();
    if(st)
    {
        PRINT_ERROR("%d", "m_eventThread->Start() returned %d", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Shutdown(std::string saveToFilename)
{
    dcgmReturn_t dcgmSt, retSt = DCGM_ST_OK;
    int st;
    lwmlVgpuInstance_t vgpuInstanceCount;

    if(m_eventThread)
    {
        PRINT_INFO("", "Stopping event thread.");
        st = m_eventThread->StopAndWait(10000);
        if(st)
        {
            PRINT_WARNING("", "Killing event thread that is still running.");
            m_eventThread->Kill();
        }
        else
            PRINT_INFO("", "Event thread was stopped normally.");
        delete m_eventThread;
        m_eventThread = 0;
    }
    else
        PRINT_WARNING("", "m_eventThread was NULL");

    /* Wake up the cache manager thread if it's sleeping. No need to wait */
    Stop();
    UpdateAllFields(0);

    /* Wait for the thread to exit for a reasonable amount of time. After that,
       just kill the polling thread so we don't wait forever */
    st = StopAndWait(30000);
    if(st)
    {
        PRINT_WARNING("", "Killing stats thread that is still running.");
        Kill();
    }

    /* Save stats if the user requested it */
    if(saveToFilename.length() > 0)
    {
        st = SaveCache(saveToFilename);
        if(st != DCGM_ST_OK && st != DCGM_ST_NOT_SUPPORTED)
        {
            PRINT_ERROR("%s %d", "SaveCache(%s) returned %d", saveToFilename.c_str(), st);
            retSt = DCGM_ST_GENERIC_ERROR;
        }
    }

    /* Sending an empty vgpuList to free vgpuList of all GPUs*/
    vgpuInstanceCount = 0;
    for(unsigned int i = 0; i < m_numGpus; i++)
    {
        ManageVgpuList(m_gpus[i].gpuId, &vgpuInstanceCount);
    }

    if(m_entityWatchHashTable)
    {
        hashtable_destroy(m_entityWatchHashTable);
        m_entityWatchHashTable = 0;
    }

    return retSt;
}

/*****************************************************************************/
dcgmGpuBrandType_t DcgmCacheManager::GetGpuBrand(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_GPU_BRAND_UNKNOWN;
    
    return m_gpus[gpuId].brand;
}

/*****************************************************************************/
void DcgmCacheManager::EntityIdToWatchKey(dcgmcm_entity_key_t *watchKey,
                                          dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          unsigned int fieldId)
{
    if(!watchKey)
        return;

    watchKey->entityId = entityId;
    watchKey->entityGroupId = entityGroupId;
    watchKey->fieldId = fieldId;
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::AllocWatchInfo(dcgmcm_entity_key_t entityKey)
{
    dcgmcm_watch_info_p retInfo = new dcgmcm_watch_info_t;
    if(!retInfo)
    {
        PRINT_CRITICAL("", "Out of memory");
        return 0;
    }
    
    retInfo->watchKey = entityKey;
    retInfo->isWatched = 0;
    retInfo->hasSubscribedWatchers = 0;
    retInfo->lastStatus = LWML_SUCCESS;
    retInfo->lastQueriedUsec = 0;
    retInfo->monitorFrequencyUsec = 0;
    retInfo->maxAgeUsec = 0;
    retInfo->execTimeUsec = 0;
    retInfo->fetchCount = 0;
    retInfo->timeSeries = 0;
    return retInfo;
}

/*****************************************************************************/
void DcgmCacheManager::FreeWatchInfo(dcgmcm_watch_info_p watchInfo)
{
    /* Call the static version that is used by the hashtable callbacks */
    entityValueFreeCB(watchInfo);
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::GetEntityWatchInfo(dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         unsigned int fieldId, int createIfNotExists)
{
    dcgmcm_watch_info_p retInfo = 0;
    dcgmMutexReturn_t mutexReturn;
    void *hashKey = 0;
    int st; 

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Global watches have no entityId */
    if(entityGroupId == DCGM_FE_NONE)
        entityId = 0;

    EntityIdToWatchKey((dcgmcm_entity_key_t *)&hashKey, entityGroupId, entityId, fieldId);

    retInfo = (dcgmcm_watch_info_p)hashtable_get(m_entityWatchHashTable, hashKey);
    if(!retInfo)
    {
        if(!createIfNotExists)
        {
            if(mutexReturn == DCGM_MUTEX_ST_OK)
                dcgm_mutex_unlock(m_mutex);
            PRINT_DEBUG("%u %u %u", "watch key eg %u, eid %u, fieldId %u doesn't exist. createIfNotExists == false",
                        entityGroupId, entityId, fieldId);
            return NULL;
        }

        /* Allocate a new one */
        PRINT_DEBUG("%p %u %u %u", "Adding WatchInfo on entityKey %p (eg %u, entityId %u, fieldId %u)",
                    hashKey, entityGroupId, entityId, fieldId);
        dcgmcm_entity_key_t addKey;
        EntityIdToWatchKey(&addKey, entityGroupId, entityId, fieldId);
        retInfo = AllocWatchInfo(addKey);
        st = hashtable_set(m_entityWatchHashTable, hashKey, retInfo);
        if(st)
        {
            PRINT_ERROR("%d", "hashtable_set failed with st %d. Likely out of memory", st);
            FreeWatchInfo(retInfo);
            retInfo = 0;
        }
    }

    if(mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    return retInfo;
}

/*****************************************************************************/
int DcgmCacheManager::HasAccountingPidBeenSeen(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key, *elem;
    int st;
    kv_lwrsor_t cursor;

    key.pid = pid;
    key.timestamp = timestamp;

    elem = (dcgmcm_pid_seen_p)keyedvector_find_by_key(m_accountingPidsSeen, &key, KV_LGE_EQUAL, &cursor);
    if(elem)
    {
        PRINT_DEBUG("%u %lld", "PID %u, ts %lld FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 1;
    }
    else
    {
        PRINT_DEBUG("%u %lld", "PID %u, ts %lld NOT FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheAccountingPid(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key;
    int st;
    kv_lwrsor_t cursor;

    key.pid = pid;
    key.timestamp = timestamp;

    st = keyedvector_insert(m_accountingPidsSeen, &key, &cursor);
    if(st)
    {
        PRINT_ERROR("%d %u %lld", "Error %d from keyedvector_insert pid %u, timestamp %lld",
                    st, key.pid, (long long)key.timestamp);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::EmptyAccountingPidCache(void)
{
    PRINT_DEBUG("", "Pid seen cache emptied");
    keyedvector_empty(m_accountingPidsSeen);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::EmptyCache(void)
{
    dcgm_mutex_lock(m_mutex);
    ClearAllEntities(1);
    EmptyAccountingPidCache();
    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::LoadCache(std::string filename)
{
    PRINT_DEBUG("%s", "Loading cache not supported. Tried from %s", filename.c_str());
    return DCGM_ST_NOT_SUPPORTED;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SaveCache(std::string filename)
{

    PRINT_DEBUG("%s", "Saving cache not supported. Tried to %s", filename.c_str());
    return DCGM_ST_NOT_SUPPORTED;
}

/*********************************f********************************************/
int DcgmCacheManager::GpuIdToLwmlIndex(unsigned int gpuId)
{
    /* Treat as index for now. Just bounds check it */
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;
    else
        return m_gpus[gpuId].lwmlIndex;
}

/*********************************f********************************************/
unsigned int DcgmCacheManager::LwmlIndexToGpuId(int lwmlIndex)
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].lwmlIndex == (unsigned int)lwmlIndex)
            return m_gpus[i].gpuId;
    }

    PRINT_ERROR("%d %d", "lwmlIndex %d not found in %u gpus", lwmlIndex, m_numGpus);
    return 0;
}

/*********************************f********************************************/
dcgmReturn_t DcgmCacheManager::Start(void)
{
    int st = LwcmThread::Start();

    if(st)
        return DCGM_ST_GENERIC_ERROR;
    else
        return DCGM_ST_OK;
}

/*********************************f********************************************/
DcgmcmGpuStatus_t DcgmCacheManager::GetGpuStatus(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DcgmcmGpuStatusUnknown;

    return m_gpus[gpuId].status;
}

/******************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuArch(unsigned int gpuId,
                                          lwmlChipArchitecture_t &arch)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    arch =  m_gpus[gpuId].arch;

    return DCGM_ST_OK;
}

/*********************************f********************************************/
dcgmReturn_t DcgmCacheManager::PauseGpu(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    switch (m_gpus[gpuId].status)
    {
        case DcgmcmGpuStatusDisabled:
        case DcgmcmGpuStatusUnknown:
        default:
            /* Nothing to do */
            return DCGM_ST_OK;

        case DcgmcmGpuStatusOk:
            /* Pause the GPU */
            PRINT_INFO("%d", "gpuId %d PAUSED.", gpuId);
            m_gpus[gpuId].status = DcgmcmGpuStatusDisabled;
            /* Force an update to occur so that we get blank values saved */
            (void)UpdateAllFields(1);
            return DCGM_ST_OK;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ResumeGpu(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    switch(m_gpus[gpuId].status)
    {
        case DcgmcmGpuStatusOk:
        case DcgmcmGpuStatusUnknown:
        default:
            /* Nothing to do */
            return DCGM_ST_OK;

        case DcgmcmGpuStatusDisabled:
            /* Pause the GPU */
            PRINT_INFO("%d", "gpuId %d RESUMED.", gpuId);
            m_gpus[gpuId].status = DcgmcmGpuStatusOk;
            return DCGM_ST_OK;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Pause()
{
    for(unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status > DcgmcmGpuStatusUnknown && m_gpus[i].status != DcgmcmGpuStatusDetached)
            PauseGpu(m_gpus[i].gpuId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Resume()
{
    for(unsigned int i = 0; i < m_numGpus; i++)
    {
        if(m_gpus[i].status > DcgmcmGpuStatusUnknown && m_gpus[i].status != DcgmcmGpuStatusDetached)
            ResumeGpu(m_gpus[i].gpuId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateAllFields(int waitForUpdate)
{
    int st;
    long long waitForFinishedCycle = 0;
    unsigned int sleepAtATimeMs = 1000;

    dcgm_mutex_lock(m_mutex);
    /*
     * Which cycle should we wait for? If one is in progress, wait for the one
     * after the current. This can be simplified as wait for the next cycle
     * that starts to finish
     */
    waitForFinishedCycle = m_runStats.updateCycleStarted + 1;

    /* Other UpdateAllFields could be waiting on this cycle as well, but that's ok.
     * They would have had to get the lock in between us Unlock()ing below and the
     * polling loop getting the lock. Either way, we're consistent thanks to the lock */
    m_runStats.shouldFinishCycle = DCGM_MAX(waitForFinishedCycle, m_runStats.shouldFinishCycle);

    lwosCondSignal(&m_startUpdateCondition);
    dcgm_mutex_unlock(m_mutex);
    // Add some kind of incrementing here.

    if(!waitForUpdate)
        return DCGM_ST_OK; /* We don't care when it finishes. Just return */

    /* Wait for signals that update loops have completed until the loop we care
       about has completed */
    while(m_runStats.updateCycleFinished < waitForFinishedCycle)
    {
#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%u %lld %lld", "Sleeping %u ms. %lld < %lld",
                    sleepAtATimeMs, m_runStats.updateCycleFinished, waitForFinishedCycle);
#endif
        dcgm_mutex_lock(m_mutex);

        /* Check the updateCycleFinished one more time, now that we got the lock */
        if(m_runStats.updateCycleFinished < waitForFinishedCycle)
        {
            st = m_mutex->CondWait(&m_updateCompleteCondition, sleepAtATimeMs);
#ifdef DEBUG_UPDATE_LOOP
            PRINT_DEBUG("%d", "UpdateAllFields() RETURN st %d", st);
        }
        else
        {
            PRINT_DEBUG("", "UpdateAllFields() skipped lwosCondWait()");
#endif
        }

        dcgm_mutex_unlock(m_mutex);

#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%d %lld %lld", "Woke up to st %d. updateCycleFinished %lld, waitForFinishedCycle %lld",
                    st, m_runStats.updateCycleFinished, waitForFinishedCycle);
#endif

        /* Make sure we don't get stuck waiting when a shutdown is requested */
        if(ShouldStop())
            return DCGM_ST_OK;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ManageDeviceEvents(unsigned int addWatchOnGpuId,
                                                  unsigned short addWatchOnFieldId)
{
    unsigned long long desiredEvents[DCGM_MAX_NUM_DEVICES] = {0};
    int somethingChanged;
    unsigned int gpuId;
    lwmlReturn_t lwmlReturn;
    dcgmcm_watch_info_p watchInfo = 0;
    lwmlDevice_t lwmlDevice = 0;
    dcgmReturn_t ret = DCGM_ST_OK;

    /* First, walk all GPUs to build a bitmask of which event types we care about */
    somethingChanged = 0;
    for(gpuId = 0; gpuId < m_numGpus; gpuId++)
    {
        if (m_gpus[gpuId].status == DcgmcmGpuStatusDetached)
            continue;

        /* We always subscribe for XIDs so that we have SOMETHING watched. Otherwise,
           the event thread won't start */
#if 1
        desiredEvents[gpuId] |= lwmlEventTypeXidCriticalError;
#else
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_XID_ERRORS);
        if(watchInfo->isWatched ||
           ((gpuId == addWatchOnGpuId) && (addWatchOnFieldId == watchInfo->fieldId)))
        {
            PRINT_DEBUG("%u", "gpuId %u wants lwmlEventTypeXidCriticalError", gpuId);
            desiredEvents[gpuId] |= lwmlEventTypeXidCriticalError;
        }
#endif

        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_LWLINK_ERRORS, 1);
        if(watchInfo->isWatched ||
           ((gpuId == addWatchOnGpuId) && (addWatchOnFieldId == watchInfo->watchKey.fieldId)))
        {
            PRINT_DEBUG("%u", "gpuId %u wants lwmlEventTypeLWLinkRecoveryError", gpuId);
            desiredEvents[gpuId] |= lwmlEventTypeLWLinkRecoveryError;
            PRINT_DEBUG("%u", "gpuId %u wants lwmlEventTypeLWLinkFatalError", gpuId);
            desiredEvents[gpuId] |= lwmlEventTypeLWLinkFatalError;
        }

        if(desiredEvents[gpuId])
        {
            PRINT_DEBUG("%u %llX %llX", "gpuId %u, desiredEvents x%llX, m_lwrrentEventMask x%llX",
                        gpuId, desiredEvents[gpuId], m_lwrrentEventMask[gpuId]);
        }

        if(desiredEvents[gpuId] != m_lwrrentEventMask[gpuId])
            somethingChanged = 1;
    }


    if(!somethingChanged)
        return DCGM_ST_OK; /* Nothing to do */

    ret = InitializeLwmlEventSet();
    if (ret != DCGM_ST_OK)
        return ret;

    for (gpuId = 0; gpuId < m_numGpus; gpuId++)
    {
        if (m_gpus[gpuId].status == DcgmcmGpuStatusDetached)
            continue;

        /* Did this GPU change? */
        if(desiredEvents[gpuId] == m_lwrrentEventMask[gpuId])
            continue;

        lwmlReturn = lwmlDeviceGetHandleByIndex(GpuIdToLwmlIndex(gpuId), &lwmlDevice);
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "ManageDeviceEvents: lwmlDeviceGetHandleByIndex returned %d for gpuId %d",
                        (int)lwmlReturn, (int)gpuId);
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }

        lwmlReturn = lwmlDeviceRegisterEvents(lwmlDevice, desiredEvents[gpuId], m_lwmlEventSet);
        if (lwmlReturn == LWML_ERROR_NOT_SUPPORTED)
        {
            PRINT_WARNING("%d, %llu",
                "ManageDeviceEvents: Desired events are not supported for gpuId: %d. Events mask: %llu",
                (int)gpuId,
                desiredEvents[gpuId]);
            continue;
        }
        else if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d",
                "ManageDeviceEvents: lwmlDeviceRegisterEvents returned %d for gpuId %d",
                (int)lwmlReturn,
                (int)gpuId);
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }

        PRINT_DEBUG("%d %llX", "Set lwmlIndex %d event mask to x%llX", gpuId,
                    desiredEvents[gpuId]);

        m_lwrrentEventMask[gpuId] = desiredEvents[gpuId];
    }

    return DCGM_ST_OK;
}

/**
 * \brief Checks if used lwml version has fixed lwlink bandwidth functionality
 *
 * \param[out] outParsedVersion If not null, will contain parsed version on the lwml library
 *
 * \return True if lwlink bandwidth functionality is fixed; False otherwise
 *
 * \note Will return True if lwml version contains anything but digits.
 *       That means build from dev branches will always return True.
 */
static bool IsLwLinkFixedVersion(std::string *outParsedVersion = NULL)
{
    /* LwLink bandwidth issue was fixed in TRD418_98 branch. Released driver version: 418.40.03
     * See http://lwbugs/2339936 and
     * https://confluence.lwpu.com/display/CSSRM/Bug+2441268+-+lwml+lwlink+APIs+return+incorrect+byte+counts
     * for details
     */
    static const std::string minSupportedVersion("4184003");
    char driverVersion[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
    
    lwmlReturn_t lwmlReturn = lwmlSystemGetDriverVersion(driverVersion, sizeof(driverVersion));
    if (LWML_SUCCESS != lwmlReturn)
    {
        PRINT_ERROR("%d %s",
                    "lwmlSystemGetDriverVersion returned %d: %s. Rolling back to bytes",
                    (int)lwmlReturn,
                    lwmlErrorString(lwmlReturn));
        return false;
    }

    std::string version(driverVersion);
    version.erase(std::remove(version.begin(), version.end(), '.'), version.end());
    if (version.empty())
    {
        PRINT_DEBUG("", "lwmlSystemGetDriverVersion returned an empty string.");
        return false;
    }

    if (outParsedVersion != NULL)
    {
        (*outParsedVersion).assign(version);
    }

    std::string::const_iterator it  = version.begin();
    std::string::const_iterator end = version.end();
    for (; it != end; ++it)
    {
        if ('0' <= *it && *it <= '9')
        {
            continue;
        }

        /* Assuming that if version has non numeric symbols, we are using a build from some dev branch */
        return true;
    }

    /* This check will work until driver version 1000. Given that we've gone from 334 to 318 in 5 years, 
       we will no longer need this WaR by the time we hit driver version 1000 */
    return version >= minSupportedVersion;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::LwmlPreWatchLwLink(unsigned int gpuId, unsigned int link)
{
    lwmlReturn_t lwmlReturn;
    /* Set LwLink bandwidth units to bytes if not already bytes */
    lwmlLwLinkUtilizationControl_t control;

    if (link >= DCGM_LWLINK_MAX_LINKS_PER_GPU)
    {
        PRINT_ERROR("%u", "Invalid link %u", link);
        return DCGM_ST_GENERIC_ERROR;
    }

    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u %u", "LwmlPreWatchLwLink: invalid gpu id %u with %u GPUs detected.", gpuId, m_numGpus);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Make sure the link states are up to date before we check them */
    UpdateLwLinkLinkState(gpuId);

    if (m_gpus[gpuId].lwLinkLinkState[link] != DcgmLwLinkLinkStateUp)
    {
        PRINT_DEBUG("%u %u %d",
                    "Skipping Link initialization for disabled linkId %u of gpuId %u in state %d",
                    link,
                    gpuId,
                    (int)m_gpus[gpuId].lwLinkLinkState[link]);
        return DCGM_ST_OK; /* Return OK here so that adding watches won't return an error */
    }


    const int LWML_LWLINK_COUNTER_UNIT_RESERVED = 3; /* TODO: Remove this once this enum is available in lwml.10.0.h */

    memset(&control, 0, sizeof(control));

    int reset         = 1;
    control.units     = (lwmlLwLinkUtilizationCountUnits_t)LWML_LWLINK_COUNTER_UNIT_BYTES;
    control.pktfilter = LWML_LWLINK_COUNTER_PKTFILTER_ALL;

    std::string driverVersion;
    if (IsLwLinkFixedVersion(&driverVersion))
    {
        PRINT_INFO("Driver version: %s",
                   "Driver version with fixed LwLink bandwidth detected: %s. Will use flits instead of bytes",
                   driverVersion.c_str());
        control.units = (lwmlLwLinkUtilizationCountUnits_t)LWML_LWLINK_COUNTER_UNIT_RESERVED;
    }
    else
    {
        PRINT_WARNING("Driver with a known LwLink bandwith issue: %s",
                      "Driver with a known LwLink bandwith issue: %s. Rolling back to bytes",
                      driverVersion.c_str());
    }

    /* Try setting unit FLITS first. If this fails with ILWALID_ARGUMENT, fall back to bytes
       See http://lwbugs/2339936 and
       https://confluence.lwpu.com/display/CSSRM/Bug+2441268+-+lwml+lwlink+APIs+return+incorrect+byte+counts
       for details */
    lwmlReturn = lwmlDeviceSetLwLinkUtilizationControl(
        m_gpus[gpuId].lwmlDevice, link, DCGMCM_LWLINK_COUNTER_BYTES, &control, reset);
    if (lwmlReturn == LWML_ERROR_ILWALID_ARGUMENT)
    {
        control.units     = (lwmlLwLinkUtilizationCountUnits_t)LWML_LWLINK_COUNTER_UNIT_BYTES;
        control.pktfilter = LWML_LWLINK_COUNTER_PKTFILTER_ALL;

        lwmlReturn = lwmlDeviceSetLwLinkUtilizationControl(
            m_gpus[gpuId].lwmlDevice, link, DCGMCM_LWLINK_COUNTER_BYTES, &control, reset);
    }

    if (lwmlReturn != LWML_SUCCESS)
    {
        PRINT_ERROR("%d %d %s %u",
                    "lwmlDeviceSetLwLinkUtilizationControl returned %d for %d: %s (control.units = %u)",
                    (int)lwmlReturn,
                    link,
                    lwmlErrorString(lwmlReturn),
                    control.units);
        return DCGM_ST_LWML_ERROR;
    }

    if (control.units == LWML_LWLINK_COUNTER_UNIT_RESERVED)
    {
        m_gpus[gpuId].lwLinkCountersAreFlits = true;
    }
    else
    {
        m_gpus[gpuId].lwLinkCountersAreFlits = false;
    }

    PRINT_DEBUG("%d %d",
                "lwmlDeviceSetLwLinkUtilization successful for link %d. isFlits %d",
                link,
                (int)m_gpus[gpuId].lwLinkCountersAreFlits);


    /* Unfreeze the counters - Volta LwLink counters are frozen by default */
    lwmlReturn = lwmlDeviceFreezeLwLinkUtilizationCounter(
        m_gpus[gpuId].lwmlDevice, link, DCGMCM_LWLINK_COUNTER_BYTES, LWML_FEATURE_DISABLED);
    if (lwmlReturn != LWML_SUCCESS)
    {
        PRINT_ERROR("%d %u %u",
                    "Error %d from lwmlDeviceFreezeLwLinkUtilizationControl for gpuId %u, link %u",
                    (int)lwmlReturn,
                    gpuId,
                    link);
        return DCGM_ST_LWML_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::LwmlPreWatch(unsigned int gpuId, unsigned short dcgmFieldId)
{
    lwmlReturn_t lwmlReturn;
    lwmlDevice_t lwmlDevice = 0;
    dcgm_field_meta_p fieldMeta = 0;
    lwmlEnableState_t enabledState;
    dcgmReturn_t dcgmSt;

    if (gpuId != (unsigned int)-1 && gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u %u", "LwmlPreWatch: gpuId %u too high. We've detected %u GPUs",
                    gpuId, m_numGpus);
        return DCGM_ST_GENERIC_ERROR;
    }

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        if (m_gpus[gpuId].status == DcgmcmGpuStatusFakeGpu)
        {
            PRINT_DEBUG("%u %u", "Skipping LwmlPreWatch for fieldId %u, fake gpuId %u", dcgmFieldId, gpuId);
            return DCGM_ST_OK;
        }

        lwmlReturn = lwmlDeviceGetHandleByIndex(m_gpus[gpuId].lwmlIndex, &lwmlDevice);
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %u", "LwmlPreWatch: lwmlDeviceGetHandleByIndex returned %d for gpuId %u",
                        (int)lwmlReturn, gpuId);
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }
    }

    switch(dcgmFieldId)
    {
        case DCGM_FI_DEV_ACCOUNTING_DATA:
            lwmlReturn = lwmlDeviceGetAccountingMode(lwmlDevice, &enabledState);
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %d", "lwmlDeviceGetAccountingMode returned %d for gpuId %u",
                            (int)lwmlReturn, gpuId);
                return DCGM_ST_LWML_ERROR;
            }
            if(enabledState == LWML_FEATURE_ENABLED)
            {
                PRINT_DEBUG("%u", "Accounting is already enabled for gpuId %u", gpuId);
                break;
            }

            /* Enable accounting */
            lwmlReturn = lwmlDeviceSetAccountingMode(lwmlDevice, LWML_FEATURE_ENABLED);
            if(lwmlReturn == LWML_ERROR_NO_PERMISSION)
            {
                PRINT_DEBUG("%d", "lwmlDeviceSetAccountingMode() got no permission. running as uid %d",
                            geteuid());
                return DCGM_ST_REQUIRES_ROOT;
            }
            else if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "lwmlDeviceSetAccountingMode returned %d for gpuId %u",
                            (int)lwmlReturn, gpuId);
                return DCGM_ST_LWML_ERROR;
            }

            PRINT_DEBUG("%u", "lwmlDeviceSetAccountingMode successful for gpuId %u", gpuId);
            break;

        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_LWLINK_ERRORS:
            ManageDeviceEvents(gpuId, dcgmFieldId);
            break;

        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L0:
        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L1:
        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L2:
        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L3:
        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L4:
        case DCGM_FI_DEV_LWLINK_BANDWIDTH_L5:
        {
            unsigned int link = dcgmFieldId - DCGM_FI_DEV_LWLINK_BANDWIDTH_L0;
            return LwmlPreWatchLwLink(gpuId, link);
        }

        case DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL:
        {
            unsigned int link;

            for(link = 0; link < DCGM_LWLINK_MAX_LINKS_PER_GPU; link++)
                LwmlPreWatchLwLink(gpuId, link);
            break;
        }

        default:
            /* Nothing to do */
            break;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::LwmlPostWatch(unsigned int gpuId, unsigned short dcgmFieldId)
{
    switch(dcgmFieldId)
    {
        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_LWLINK_ERRORS:
            ManageDeviceEvents(DCGM_GPU_ID_BAD, 0);
            break;

        default:
            /* Nothing to do */
            break;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddFieldWatch(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                             unsigned short dcgmFieldId, timelib64_t monitorFrequencyUsec,
                                             double maxSampleAge, int maxKeepSamples, DcgmWatcher watcher,
                                             bool subscribeForUpdates)
{
    dcgm_field_meta_p fieldMeta = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if(fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing global field watch to be correct scope.");
        entityGroupId = DCGM_FE_NONE;
    }

    /* Trigger the update loop to buffer updates from now on */
    if(subscribeForUpdates)
        m_haveAnyLiveSubscribers = true;

    if (entityGroupId != DCGM_FE_NONE)
    {
        return AddEntityFieldWatch(entityGroupId, entityId, dcgmFieldId, monitorFrequencyUsec, 
                                   maxSampleAge, maxKeepSamples, watcher, subscribeForUpdates);
    }
    else
    {
        return AddGlobalFieldWatch(dcgmFieldId, monitorFrequencyUsec, maxSampleAge, 
                                   maxKeepSamples, watcher, subscribeForUpdates);
    }    
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetFieldWatchFreq(unsigned int gpuId, unsigned short fieldId, timelib64_t *freqUsec)
{
    dcgmcm_watch_info_p watchInfo = 0;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "%u is not a valid field ID", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope == DCGM_FS_DEVICE)
    {
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }

    *freqUsec = 0;
    
    if(watchInfo)
        *freqUsec = watchInfo->monitorFrequencyUsec;

    return DCGM_ST_OK;
}

/*****************************************************************************/
timelib64_t DcgmCacheManager::GetMaxAgeUsec(timelib64_t monitorFrequencyUsec, double maxAgeSeconds, int maxKeepSamples)
{
    timelib64_t maxAgeUsec;
    timelib64_t maxKeepSamplesInUsec;

    /* Either value can be 0, which is null */

    if(maxAgeSeconds == 0.0)
        return (timelib64_t)maxKeepSamples * monitorFrequencyUsec;
    
    maxAgeUsec = (timelib64_t)(maxAgeSeconds * 1000000.0);
    
    if(!maxKeepSamples)
        return maxAgeUsec;
    
    /* Both values are nonzero. Use the most restrictive of the two */
    maxKeepSamplesInUsec = (timelib64_t)maxKeepSamples * monitorFrequencyUsec;

    return DCGM_MIN(maxKeepSamplesInUsec, maxAgeUsec);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFieldWatch(dcgmcm_watch_info_p watchInfo, timelib64_t monitorFrequencyUsec, double maxAgeSec, 
                                                int maxKeepSamples, DcgmWatcher watcher)
{
    if(!watchInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    if(!watchInfo->isWatched)
    {
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_NOT_WATCHED;
    }

    watchInfo->monitorFrequencyUsec = monitorFrequencyUsec;
    watchInfo->maxAgeUsec = GetMaxAgeUsec(monitorFrequencyUsec, maxAgeSec, maxKeepSamples);

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGpuFieldWatched(unsigned int gpuId, unsigned short dcgmFieldId,
                                                 bool *isWatched)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_DEVICE) {
        PRINT_ERROR("%u %d %d", "field ID %u has scope %d but this function only works for DEVICE (%d) scope fields",
                    dcgmFieldId, fieldMeta->scope, DCGM_FS_DEVICE);
        return DCGM_ST_BADPARAM;
    }

    *isWatched = false;
    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if(watchInfo)
        *isWatched = watchInfo->isWatched;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGpuFieldWatchedOnAnyGpu(unsigned short fieldId, bool *isWatched)
{
    dcgmReturn_t st;

    if (isWatched == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    std::vector<unsigned int> gpuIds;
    this->GetGpuIds(1, gpuIds);

    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        unsigned int gpuId = gpuIds.at(i);
        st = IsGpuFieldWatched(gpuId, fieldId, isWatched);
        if (DCGM_ST_OK != st)
            return st;

        if (*isWatched == true)
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGlobalFieldWatched(unsigned short dcgmFieldId, bool *isWatched)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL) {
        PRINT_ERROR("%u %d %d", "field ID %u has scope %d but this function only works for GLOBAL (%d) scope fields",
                    dcgmFieldId, fieldMeta->scope, DCGM_FS_GLOBAL);
        return DCGM_ST_BADPARAM;
    }

    *isWatched = false;
    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);
    if(watchInfo)
        *isWatched = watchInfo->isWatched;
    return DCGM_ST_OK;
}

bool DcgmCacheManager::AnyGlobalFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;

    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_GLOBAL)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGlobalFieldWatched(fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (isWatched)
        {
            return true;
        }
    }

    return false;
}

bool DcgmCacheManager::AnyFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    dcgmReturn_t st = DCGM_ST_OK;
    bool isWatched = false;

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
            continue;

        if (fieldMeta->scope == DCGM_FS_GLOBAL)
        {
            st = IsGlobalFieldWatched(fieldId, &isWatched);
        }
        else if (fieldMeta->scope == DCGM_FS_DEVICE)
        {
            st = IsGpuFieldWatchedOnAnyGpu(fieldId, &isWatched);
        }

        if (DCGM_ST_OK == st && isWatched)
            return true;
    }

    return false;
}

bool DcgmCacheManager::AnyGpuFieldsWatchedAnywhere(std::vector<unsigned short> *fieldIds)
{
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t st = GetGpuIds(1, gpuIds);
    if (DCGM_ST_OK != st)
        return false;

    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        unsigned int gpuId = gpuIds.at(i);
        if (AnyGpuFieldsWatched(gpuId, fieldIds))
        {
            return true;
        }
    }

    return false;
}

bool DcgmCacheManager::AnyGpuFieldsWatched(unsigned int gpuId, std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;

    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_DEVICE)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGpuFieldWatched(gpuId, fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (isWatched)
        {
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveFieldWatch(dcgm_field_entity_group_t entityGroupId, unsigned int entityId,
                                                unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher)
{
    dcgm_field_meta_p fieldMeta = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD; 

    if (entityGroupId != DCGM_FE_NONE)
    {
        return RemoveEntityFieldWatch(entityGroupId, entityId, dcgmFieldId, clearCache, watcher);
    }
    else
    {
        return RemoveGlobalFieldWatch(dcgmFieldId, clearCache, watcher);
    }      
}


/*************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveWatcher(dcgmcm_watch_info_p watchInfo,
                                             dcgm_watch_watcher_info_t *watcher,
                                             int clearCache)
{
    std::vector<dcgm_watch_watcher_info_t>::iterator it;
    
    for(it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if((*it).watcher == watcher->watcher)
        {
            PRINT_DEBUG("%u %u", "RemoveWatcher removing existing watcher type %u, connectionId %u", 
                        watcher->watcher.watcherType, watcher->watcher.connectionId);

            watchInfo->watchers.erase(it);
            /* Update the watchInfo frequency and quota now that we removed a watcher */
            UpdateWatchFromWatchers(watchInfo);

            /* Last watcher? */
            if(watchInfo->watchers.size() < 1)
            {
                watchInfo->isWatched = 0;                
                
                if(watchInfo->watchKey.entityGroupId == DCGM_FE_GPU)
                {
                    LwmlPostWatch(GpuIdToLwmlIndex(watchInfo->watchKey.entityId), 
                                  watchInfo->watchKey.fieldId);
                }
                else if(watchInfo->watchKey.entityGroupId == DCGM_FE_NONE)
                    LwmlPostWatch(-1, watchInfo->watchKey.fieldId);

                /* If requested, clear the cache once the last watcher goes away */
                if(clearCache && watchInfo->timeSeries)
                {
                    timeseries_destroy(watchInfo->timeSeries);
                    watchInfo->timeSeries = 0;
                }
            }

            return DCGM_ST_OK;
        }
    }

    PRINT_DEBUG("%u %u", "RemoveWatcher() type %u, connectionId %u was not a watcher", 
                watcher->watcher.watcherType, watcher->watcher.connectionId);
    return DCGM_ST_NOT_WATCHED;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddOrUpdateWatcher(dcgmcm_watch_info_p watchInfo, 
                                                  bool *wasAdded,
                                                  dcgm_watch_watcher_info_t *newWatcher)
{
    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    for(it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if((*it).watcher == newWatcher->watcher)
        {
            PRINT_DEBUG("%u %u", "Updating existing watcher type %u, connectionId %u", 
                        newWatcher->watcher.watcherType, newWatcher->watcher.connectionId);

            *it = *newWatcher;
            *wasAdded = false;
            /* Update the watchInfo frequency and quota now that we updated a watcher */
            UpdateWatchFromWatchers(watchInfo);
            return DCGM_ST_OK;
        }
    }

    PRINT_DEBUG("%u %u", "Adding new watcher type %u, connectionId %u", 
                newWatcher->watcher.watcherType, newWatcher->watcher.connectionId);

    watchInfo->watchers.push_back(*newWatcher);
    *wasAdded = true;

    /* Update the watchInfo frequency and quota now that we added a watcher */
    UpdateWatchFromWatchers(watchInfo);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateWatchFromWatchers(dcgmcm_watch_info_p watchInfo)
{
    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    it = watchInfo->watchers.begin();

    if(it == watchInfo->watchers.end())
    {
        watchInfo->hasSubscribedWatchers = 0;
        return DCGM_ST_NOT_WATCHED;
    }

    /* Don't update watchInfo's value here because we don't want non-locking readers to them in a temporary state */
    timelib64_t minMonitorFreqUsec = it->monitorFrequencyUsec;
    timelib64_t minMaxAgeUsec = it->maxAgeUsec;
    bool hasSubscribedWatchers = it->isSubscribed; 

    for(++it; it != watchInfo->watchers.end(); ++it)
    {
        minMonitorFreqUsec = DCGM_MIN(minMonitorFreqUsec, it->monitorFrequencyUsec);
        minMaxAgeUsec = DCGM_MIN(minMaxAgeUsec, it->maxAgeUsec);
        if(it->isSubscribed)
           hasSubscribedWatchers = 1;
    }

    watchInfo->monitorFrequencyUsec = minMonitorFreqUsec;
    watchInfo->maxAgeUsec = minMaxAgeUsec;
    watchInfo->hasSubscribedWatchers = hasSubscribedWatchers;

    PRINT_DEBUG("%lld %lld %d", "UpdateWatchFromWatchers minMonitorFreqUsec %lld, minMaxAgeUsec %lld, hsw %d", 
                (long long)minMonitorFreqUsec, (long long)minMaxAgeUsec, watchInfo->hasSubscribedWatchers);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddEntityFieldWatch(dcgm_field_entity_group_t entityGroupId, unsigned int entityId,
                                                   unsigned short dcgmFieldId,
                                                   timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                                   int maxKeepSamples, DcgmWatcher watcher, bool subscribeForUpdates)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    bool wasAdded = false;
    dcgm_watch_watcher_info_t newWatcher;

    if(dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
       watcher table */
    newWatcher.watcher = watcher;
    newWatcher.monitorFrequencyUsec = monitorFrequencyUsec;
    newWatcher.maxAgeUsec = GetMaxAgeUsec(monitorFrequencyUsec, maxSampleAge, maxKeepSamples);
    newWatcher.isSubscribed = subscribeForUpdates ? 1 : 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 1);

    /* New watch? */
    if(!watchInfo->isWatched && entityGroupId == DCGM_FE_GPU)
    {
        watchInfo->lastQueriedUsec = 0;

        /* Do the pre-watch first in case it fails */
        dcgmReturn = LwmlPreWatch(GpuIdToLwmlIndex(entityId), dcgmFieldId);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%u %u %d", "LwmlPreWatch eg %u,  eid %u, failed with %d",
                        entityGroupId, entityId, (int)dcgmReturn);
            dcgm_mutex_unlock(m_mutex);
            return dcgmReturn;
        }
    }
    
    /* Add or update the watcher in our table */
    AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

    watchInfo->isWatched = 1;

    dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %u %u %lld %f %d %d", "AddFieldWatch eg %u, eid %u, fieldId %u, mfu %lld, msa %f, mka %d, sfu %d",
                entityGroupId, entityId, dcgmFieldId, (long long int)monitorFrequencyUsec, maxSampleAge, 
                maxKeepSamples, subscribeForUpdates ? 1 : 0);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveEntityFieldWatch(dcgm_field_entity_group_t entityGroupId, unsigned int entityId,
                                                      unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmMutexReturn_t mutexReturn;
    dcgm_watch_watcher_info_t remWatcher;
    dcgmReturn_t retSt = DCGM_ST_OK;

    if(dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;
    
    /* Populate the cache manager version of watcher so we can remove it in this watchInfo's
       watcher table */
    remWatcher.watcher = watcher;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);
    if(!watchInfo)
    {
        retSt = DCGM_ST_NOT_WATCHED;
    }
    else
        RemoveWatcher(watchInfo, &remWatcher, clearCache);

    if(mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %u %u %d", "RemoveEntityFieldWatch eg %u, eid %u, lwmlFieldId %u, clearCache %d",
            entityGroupId, entityId, dcgmFieldId, clearCache);

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddGlobalFieldWatch(unsigned short dcgmFieldId, timelib64_t monitorFrequencyUsec,
                               double maxSampleAge, int maxKeepSamples, DcgmWatcher watcher, bool subscribeForUpdates)
{
    dcgmcm_watch_info_p watchInfo;
    timelib64_t maxSampleAgeUsec;
    bool wasAdded = false;
    dcgm_watch_watcher_info_t newWatcher;

    if(dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    
    /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
        watcher table */
    newWatcher.watcher = watcher;
    newWatcher.monitorFrequencyUsec = monitorFrequencyUsec;
    newWatcher.maxAgeUsec = GetMaxAgeUsec(monitorFrequencyUsec, maxSampleAge, maxKeepSamples);
    newWatcher.isSubscribed = subscribeForUpdates;

    /* New watch? */
    if(!watchInfo->isWatched)
    {
        LwmlPreWatch(-1, dcgmFieldId);
    }

    /* Add or update the watcher in our table */
    AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

    watchInfo->isWatched = 1;

    dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %lld %f %d %d", "AddGlobalFieldWatch dcgmFieldId %u, mfu %lld, msa %f, mka %d, sfu %d",
                dcgmFieldId, (long long int)monitorFrequencyUsec, maxSampleAge, maxKeepSamples,
                subscribeForUpdates ? 1 : 0);

    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveGlobalFieldWatch(unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmMutexReturn_t mutexReturn;
    dcgm_watch_watcher_info_t remWatcher;

    if(dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;
    
    /* Populate the cache manager version of watcher so we can remove it in this watchInfo's
       watcher table */
    remWatcher.watcher = watcher;

    mutexReturn = dcgm_mutex_lock(m_mutex);

    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);

    if(watchInfo)
        RemoveWatcher(watchInfo, &remWatcher, clearCache);

    if(mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %d", "RemoveGlobalFieldWatch dcgmFieldId %u, clearCache %d",
                dcgmFieldId, clearCache);

    return DCGM_ST_OK;
}

/*****************************************************************************/
/*
 * Helper to colwert a timeseries entry to a sample
 *
 */
static dcgmReturn_t DcgmcmTimeSeriesEntryToSample(dcgmcm_sample_p sample, timeseries_entry_p entry,
                                                  timeseries_p timeseries)
{
    sample->timestamp = entry->usecSince1970;


    switch(timeseries->tsType)
    {
        case TS_TYPE_DOUBLE:
            sample->val.d = entry->val.dbl;
            sample->val2.d = entry->val2.dbl;
            break;
        case TS_TYPE_INT64:
            sample->val.i64 = entry->val.i64;
            sample->val2.i64 = entry->val2.i64;
            break;
        case TS_TYPE_STRING:
            sample->val.str = strdup((char *)entry->val.ptr);
            if(!sample->val.str)
            {
                sample->val2.ptrSize = 0;
                return DCGM_ST_MEMORY;
            }
            sample->val2.ptrSize = strlen(sample->val.str)+1;
            break;
        case TS_TYPE_BLOB:
            sample->val.blob = malloc(entry->val2.ptrSize);
            if(!sample->val.blob)
            {
                sample->val2.ptrSize = 0;
                return DCGM_ST_MEMORY;
            }
            sample->val2.ptrSize = entry->val2.ptrSize;
            memcpy(sample->val.blob, entry->val.ptr, entry->val2.ptrSize);
            break;

        default:
            PRINT_ERROR("%d", "Shouldn't get here for type %d", (int)timeseries->tsType);
            return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static 
dcgmReturn_t DcgmcmWriteTimeSeriesEntryToFvBuffer(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId,
                                                  unsigned short fieldId,
                                                  timeseries_entry_p entry, 
                                                  DcgmFvBuffer *fvBuffer, 
                                                  timeseries_p timeseries)
{
    dcgmBufferedFv_t *fv = 0; 

    switch(timeseries->tsType)
    {
        case TS_TYPE_DOUBLE:
            fv = fvBuffer->AddDoubleValue(entityGroupId, entityId, fieldId, 
                                          entry->val.dbl, entry->usecSince1970, 
                                          DCGM_ST_OK);
            break;
        case TS_TYPE_INT64:
            fv = fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 
                                         entry->val.i64, entry->usecSince1970, 
                                         DCGM_ST_OK);
            break;
        case TS_TYPE_STRING:
            fv = fvBuffer->AddStringValue(entityGroupId, entityId, fieldId, 
                                          (char *)entry->val.ptr, entry->usecSince1970, 
                                          DCGM_ST_OK);
            break;
        case TS_TYPE_BLOB:
            fv = fvBuffer->AddBlobValue(entityGroupId, entityId, fieldId, 
                                        entry->val.ptr, entry->val2.ptrSize, 
                                        entry->usecSince1970, DCGM_ST_OK);
            break;

        default:
            PRINT_ERROR("%d", "Shouldn't get here for type %d", (int)timeseries->tsType);
            return DCGM_ST_BADPARAM;
    }

    if(!fv)
    {
        PRINT_ERROR("%u %u %u", "Unexpected NULL fv returned for eg %u, eid %u, fieldId %u. Out of memory?",
                    entityGroupId, entityId, fieldId);
        return DCGM_ST_MEMORY;
    }
    
    //PRINT_DEBUG("%u %u %u %d", "eg %u, eid %u, fieldId %u buffered %d bytes.", 
    //            entityGroupId, entityId, fieldId, fv->length);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                                 unsigned int excludePid,
                                                 dcgmProcessUtilInfo_t*pidInfo, unsigned int *numPids,
                                                 timelib64_t startTime, timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;

    if(!pidInfo || !numPids)
        return DCGM_ST_BADPARAM;
    if(dcgmFieldId != DCGM_FI_DEV_GRAPHICS_PIDS && dcgmFieldId != DCGM_FI_DEV_COMPUTE_PIDS)
        return DCGM_ST_BADPARAM;

    maxPids = *numPids;
    *numPids = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_BLOB for %u. Got %d",
                    dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmRunningProcess_t *proc;
    int i, havePid;

    /* Walk forward  */
    if(startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                            KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for(; entry;
        entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if(endTime && entry->usecSince1970 > endTime)
            break;

        proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if(!proc || proc->version != dcgmRunningProcess_version)
        {
            PRINT_ERROR("", "Skipping invalid entry");
            continue;
        }

        if(excludePid && excludePid == proc->pid)
            continue; /* Skip exclusion pid */

        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for(i = 0; i < (int)(*numPids); i++)
        {
            if(proc->pid == pidInfo[i].pid)
            {
                havePid = 1;
                break;
            }
        }

        if(havePid)
            continue; /* Already have this one */

        /* We found a new PID */
        pidInfo[*numPids].pid = proc->pid;
        (*numPids)++;

        /* Have we reached our capacity? */
        if((*numPids) >= maxPids)
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    if(!(*numPids))
    {
        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                                 unsigned int excludePid,
                                                 unsigned int*pids, unsigned int *numPids,
                                                 timelib64_t startTime, timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;

    if(!pids || !numPids)
        return DCGM_ST_BADPARAM;
    if(dcgmFieldId != DCGM_FI_DEV_GRAPHICS_PIDS && dcgmFieldId != DCGM_FI_DEV_COMPUTE_PIDS)
        return DCGM_ST_BADPARAM;

    maxPids = *numPids;
    *numPids = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_BLOB for %u. Got %d",
                    dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmRunningProcess_t *proc;
    int i, havePid;

    /* Walk forward  */
    if(startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                            KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for(; entry;
        entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if(endTime && entry->usecSince1970 > endTime)
            break;

        proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if(!proc || proc->version != dcgmRunningProcess_version)
        {
            PRINT_ERROR("", "Skipping invalid entry");
            continue;
        }

        if(excludePid && excludePid == proc->pid)
            continue; /* Skip exclusion pid */

        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for(i = 0; i < (int)(*numPids); i++)
        {
            if(proc->pid == pids[i])
            {
                havePid = 1;
                break;
            }
        }

        if(havePid)
            continue; /* Already have this one */

        /* We found a new PID */
        pids[*numPids] = proc->pid;
        (*numPids)++;

        /* Have we reached our capacity? */
        if((*numPids) >= maxPids)
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    if(!(*numPids))
    {
        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PrecheckWatchInfoForSamples(dcgmcm_watch_info_p watchInfo)
{
    /* Note: This code assumes that the cache manager is locked */

    if(!watchInfo)
    {
        PRINT_DEBUG("", "PrecheckWatchInfoForSamples: not watched");
        return DCGM_ST_NOT_WATCHED;
    }

    /* Matching existing behavior: if there is data for an entity, then we can
       return it. This bypasses recent LWML failures or the field no longer
       being watched. */
    if(watchInfo->timeSeries)
        return DCGM_ST_OK;
    
    if(!watchInfo->isWatched)
    {
        PRINT_DEBUG("%u %u %u", "eg %u, eid %u, fieldId %u not watched", 
                    watchInfo->watchKey.entityGroupId, watchInfo->watchKey.entityId, 
                    watchInfo->watchKey.fieldId);
        return DCGM_ST_NOT_WATCHED;
    }
    
    if(watchInfo->lastStatus != LWML_SUCCESS)
    {
        PRINT_DEBUG("%u %u %u %u", "eg %u, eid %u, fieldId %u LWML status %u", 
                    watchInfo->watchKey.entityGroupId, watchInfo->watchKey.entityId, 
                    watchInfo->watchKey.fieldId, watchInfo->lastStatus);
        return LwmlReturnToDcgmReturn(watchInfo->lastStatus);
    }

    int numElements = timeseries_size(watchInfo->timeSeries);
    if(!numElements)
    {
        PRINT_DEBUG("%u %u %u", "eg %u, eid %u, fieldId %u has NO DATA", 
                    watchInfo->watchKey.entityGroupId, watchInfo->watchKey.entityId, 
                    watchInfo->watchKey.fieldId);
        return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidUtilLists(dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                                     unsigned int includePid,
                                                     dcgmProcessUtilSample_t*processUtilSamples,
                                                     unsigned int *numUniqueSamples,
                                                     timelib64_t startTime, timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;
    unsigned int numSamples = 0;

    if(!processUtilSamples || !numUniqueSamples)
        return DCGM_ST_BADPARAM;
    if(dcgmFieldId != DCGM_FI_DEV_GPU_UTIL_SAMPLES && dcgmFieldId != DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES)
        return DCGM_ST_BADPARAM;

    maxPids = *numUniqueSamples;
    *numUniqueSamples = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);
    
    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_DOUBLE)
    {
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_DOUBLE for %u. Got %d",
                    dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    int i, havePid;
    double utilVal;
    unsigned int pid;

    /* Walk forward  */
    if(startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                            KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for(; entry;
        entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if(endTime && entry->usecSince1970 > endTime)
            break;

        if(!entry)
        {
            PRINT_ERROR("", "Skipping invalid entry");
            continue;
        }

        utilVal = entry->val.dbl;
        pid = (unsigned int)entry->val2.dbl;

        /* Continue if it is an ilwalidPID or if the entry is not the one for which utilization is being looked for*/
        if(UINT_MAX == pid || (includePid > 0 && pid != includePid))
       {
            numSamples++;
            continue;
        }
        
        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for(i = 0; i < (int)(*numUniqueSamples); i++)
        {
            if( pid== processUtilSamples[i].pid)
            {
                havePid = 1;
                processUtilSamples[i].util += utilVal;
                numSamples++;
                break;
            }
        }

        if(havePid )
            continue; /* Already have this one */

        /* We found a new PID */
        processUtilSamples[*numUniqueSamples].pid = pid;
        processUtilSamples[*numUniqueSamples].util = utilVal;
        numSamples++;
        (*numUniqueSamples)++;

        /* Have we reached our capacity? */
        if((*numUniqueSamples) >= maxPids)
        {
            PRINT_DEBUG("%d %d", "Reached Max Capacity of ProcessSamples  - %d, maxPids = %d", *numUniqueSamples, maxPids);
            break;
        }
    }

    dcgm_mutex_unlock(m_mutex);
    /* Average utilization rates */
    for(i = 0 ; i < (int)*numUniqueSamples; i++)
    {
       processUtilSamples[i].util =  processUtilSamples[i].util/numSamples;
    }

    if(!(*numUniqueSamples))
    {
        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetLatestProcessInfo(unsigned int gpuId, unsigned int pid,
                                                    dcgmDevicePidAccountingStats_t *pidInfo)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    if(!pid || !pidInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ACCOUNTING_DATA, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        PRINT_ERROR("%d", "Expected type TS_TYPE_BLOB for DCGM_FI_DEV_ACCOUNTING_DATA. Got %d",
                    watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmDevicePidAccountingStats_t *accStats;
    dcgmDevicePidAccountingStats_t *matchingAccStats = 0;
    int Nseen = 0;

    /* Walk backwards looking for our PID */
    for(entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
        entry && !matchingAccStats;
        entry = (timeseries_entry_p)keyedvector_prev(timeseries->keyedVector, &cursor))
    {
        Nseen++;
        accStats = (dcgmDevicePidAccountingStats_t *)entry->val.ptr;
        if(!accStats)
        {
            PRINT_ERROR("", "Null entry");
            continue;
        }

        if(accStats->pid == pid)
        {
            PRINT_DEBUG("%u %d", "Found pid %u after %d entries", pid, Nseen);
            matchingAccStats = accStats;
            break;
        }
    }

    if(!Nseen || !matchingAccStats)
    {
        dcgm_mutex_unlock(m_mutex);
        PRINT_DEBUG("%u %d", "Pid %u not found after looking at %d entries", pid, Nseen);

        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    if(matchingAccStats->version != dcgmDevicePidAccountingStats_version)
    {
        dcgm_mutex_unlock(m_mutex);
        PRINT_ERROR("%d %d", "Expected accounting stats version %d. Found %d",
                    dcgmDevicePidAccountingStats_version, matchingAccStats->version);
        return DCGM_ST_GENERIC_ERROR; /* This is an internal version mismatch, not a user one */
    }

    memcpy(pidInfo, matchingAccStats, sizeof(*pidInfo));
    dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %d", "Found match for PID %u after %d records", pid, Nseen);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetInt64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   unsigned short dcgmFieldId,
                                                   int numSummaryTypes,
                                                   DcgmcmSummaryType_t *summaryTypes,
                                                   long long *summaryValues,
                                                   timelib64_t startTime, timelib64_t endTime,
                                                   pfUseEntryForSummary pfUseEntryCB, void *userData)
{
    dcgmcm_watch_info_p watchInfo = 0;
    int stIndex;

    if(!dcgmFieldId || numSummaryTypes < 1 || !summaryTypes || !summaryValues)
        return DCGM_ST_BADPARAM;

    /* Initialize all return data to blank */
    for(stIndex = 0; stIndex < numSummaryTypes; stIndex++)
    {
        summaryValues[stIndex] = DCGM_INT64_BLANK;
    }

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_INT64)
    {
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_INT64 for field %u. Got %d",
                    dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    timelib64_t prevTimestamp = 0;
    int Nseen = 0;
    long long value = 0, prevValue = 0, sumValue = 0;
    long long firstValue = DCGM_INT64_BLANK;

    /* Walk forward  */
    if(startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                            KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for(; entry;
        entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if(endTime && entry->usecSince1970 > endTime)
            break;

        
        if (pfUseEntryCB) {
            if (!pfUseEntryCB(entry, userData)) {
                continue;
            }
        }
        
        Nseen++;
        value = entry->val.i64;

        /* All of the current summary types ignore blank values */
        if(DCGM_INT64_IS_BLANK(value))
        {
            PRINT_DEBUG("%d %u", "Skipping blank value at Nseen %d. fieldId %u", 
                        Nseen, watchInfo->watchKey.fieldId);
            prevValue = value;
            prevTimestamp = entry->usecSince1970;
            continue;
        }

        /* Keep track of first non-blank value */
        if(DCGM_INT64_IS_BLANK(firstValue))
        {
            firstValue = value;
        }

        /* Keep a running sum */
        sumValue += value;

        /* Walk over each summary type the caller is requesting and do the necessary work
         * for this value */
        for(stIndex = 0; stIndex < numSummaryTypes; stIndex++)
        {
            switch(summaryTypes[stIndex])
            {
                case DcgmcmSummaryTypeMinimum:
                    if(DCGM_INT64_IS_BLANK(summaryValues[stIndex]) ||
                       value < summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeMaximum:
                    if(DCGM_INT64_IS_BLANK(summaryValues[stIndex]) ||
                                           value > summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeAverage:
                    summaryValues[stIndex] = sumValue / (long long)Nseen;
                    break;

                case DcgmcmSummaryTypeSum:
                    summaryValues[stIndex] = sumValue;
                    break;

                case DcgmcmSummaryTypeCount:
                    summaryValues[stIndex] = Nseen;
                    break;

                case DcgmcmSummaryTypeIntegral:
                {
                    timelib64_t timeDiff;
                    long long avgValue, area;

                    /* Need a time difference to callwlate an area */
                    if(!prevTimestamp)
                    {
                        summaryValues[stIndex] = 0; /* Make sure our starting value is non-blank */
                        break;
                    }

                    avgValue = (value + prevValue) / 2;
                    timeDiff = entry->usecSince1970 - prevTimestamp;
                    area = (avgValue * timeDiff);
                    summaryValues[stIndex] += area;
                    break;
                }

                case DcgmcmSummaryTypeDifference:
                {
                    summaryValues[stIndex] = value - firstValue;
                    break;
                }

                default:
                    dcgm_mutex_unlock(m_mutex);
                    PRINT_ERROR("%d", "Unhandled summaryType %d", (int)summaryTypes[stIndex]);
                    return DCGM_ST_BADPARAM;
            }

        }

        /* Save previous values before going around loop */
        prevValue = value;
        prevTimestamp = entry->usecSince1970;
    }

    dcgm_mutex_unlock(m_mutex);

    if(!Nseen)
    {

        PRINT_DEBUG("", "No values found");

        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetFp64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId,
                                                  unsigned short dcgmFieldId,
                                                  int numSummaryTypes,
                                                  DcgmcmSummaryType_t *summaryTypes,
                                                  double *summaryValues,
                                                  timelib64_t startTime, timelib64_t endTime, 
                                                  pfUseEntryForSummary pfUseEntryCB, void *userData)
{
    dcgmcm_watch_info_p watchInfo = 0;
    int stIndex;

    if(!dcgmFieldId || numSummaryTypes < 1 || !summaryTypes || !summaryValues)
        return DCGM_ST_BADPARAM;

    /* Initialize all return data to blank */
    for(stIndex = 0; stIndex < numSummaryTypes; stIndex++)
    {
        summaryValues[stIndex] = DCGM_FP64_BLANK;
    }

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if(dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if(watchInfo->timeSeries->tsType != TS_TYPE_DOUBLE)
    {
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_DOUBLE for field %u. Got %d",
                    dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;
    timelib64_t prevTimestamp = 0;
    int Nseen = 0;
    double value = 0.0, prevValue = 0.0, sumValue = 0.0;
    double firstValue = DCGM_FP64_BLANK;

    /* Walk forward  */
    if(startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                            KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for(; entry;
        entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {

        /* Past our time range? */
        if(endTime && entry->usecSince1970 > endTime)
            break;
        
        if (pfUseEntryCB) {
            if (!pfUseEntryCB(entry, userData)) {
                continue;
            }
        }
        
        Nseen++;
        value = entry->val.dbl;

        /* All of the current summary types ignore blank values */
        if(DCGM_FP64_IS_BLANK(value))
        {
            PRINT_DEBUG("%d %u", "Skipping blank value at Nseen %d. fieldId %u", 
                        Nseen, watchInfo->watchKey.fieldId);
            prevValue = value;
            prevTimestamp = entry->usecSince1970;
            continue;
        }

        /* Keep track of the first non-blank value seen */
        if(DCGM_FP64_IS_BLANK(firstValue))
        {
            firstValue = value;
        }

        /* Keep a running sum */
        sumValue += value;

        /* Walk over each summary type the caller is requesting and do the necessary work
         * for this value */
        for(stIndex = 0; stIndex < numSummaryTypes; stIndex++)
        {
            switch(summaryTypes[stIndex])
            {
                case DcgmcmSummaryTypeMinimum:
                    if(DCGM_FP64_IS_BLANK(summaryValues[stIndex]) ||
                       value < summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeMaximum:
                    if(DCGM_FP64_IS_BLANK(summaryValues[stIndex]) ||
                                           value > summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeAverage:
                    summaryValues[stIndex] = sumValue / (double)Nseen;
                    break;

                case DcgmcmSummaryTypeSum:
                    summaryValues[stIndex] = sumValue;
                    break;

                case DcgmcmSummaryTypeCount:
                    summaryValues[stIndex] = Nseen;
                    break;

                case DcgmcmSummaryTypeIntegral:
                {
                    timelib64_t timeDiff;
                    double avgValue, area;

                    /* Need a time difference to callwlate an area */
                    if(!prevTimestamp)
                    {
                        summaryValues[stIndex] = 0; /* Make sure our starting value is non-blank */
                        break;
                    }

                    avgValue = (value + prevValue) / 2.0;
                    timeDiff = entry->usecSince1970 - prevTimestamp;
                    area = (avgValue * timeDiff);
                    summaryValues[stIndex] += area;
                    break;
                }

                case DcgmcmSummaryTypeDifference:
                {
                    summaryValues[stIndex] = value - firstValue;
                    break;
                }

                default:
                    dcgm_mutex_unlock(m_mutex);
                    PRINT_ERROR("%d", "Unhandled summaryType %d", (int)summaryTypes[stIndex]);
                    return DCGM_ST_BADPARAM;
            }

        }

        /* Save previous values before going around loop */
        prevValue = value;
        prevTimestamp = entry->usecSince1970;
    }

    dcgm_mutex_unlock(m_mutex);

    if(!Nseen)
    {

        PRINT_DEBUG("", "No values found");

        if(!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetSamples(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                          dcgmcm_sample_p samples, int *Msamples,
                                          timelib64_t startTime, timelib64_t endTime,
                                          dcgmOrder_t order)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t st, retSt = DCGM_ST_OK;
    timeseries_p timeseries = 0;
    int maxSamples;
    dcgmcm_watch_info_p watchInfo = 0;

    if(!samples || !Msamples || (*Msamples)<1)
        return DCGM_ST_BADPARAM;
    if(order != DCGM_ORDER_ASCENDING && order != DCGM_ORDER_DESCENDING)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;
    
    if(fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId for global field");
        entityGroupId = DCGM_FE_NONE;
    }

    dcgm_mutex_lock(m_mutex);
    
    if (entityGroupId != DCGM_FE_NONE) {
        watchInfo = GetEntityWatchInfo(entityGroupId, entityId, fieldMeta->fieldId, 0);
    } else {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }

    maxSamples = *Msamples; /* Store the passed in value */
    *Msamples = 0; /* No samples collected yet */

    st = PrecheckWatchInfoForSamples(watchInfo);
    if(st != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return st;
    }

    /* Data type is assumed to be a time series type */

    timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;

    if(order == DCGM_ORDER_ASCENDING)
    {
        /* Which entry we start on depends on if a starting timestamp was provided or not */
        if(!startTime)
        {
            entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);
        }
        else
        {
            timeseries_entry_t key;

            key.usecSince1970 = startTime;
            entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                                KV_LGE_GREATEQUAL, &cursor);
        }

        /* Walk all samples until we fill our buffer, run out of samples, or go past our end timestamp */
        for(; entry && (*Msamples) < maxSamples;
            entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
        {
            /* Past our time range? */
            if(endTime && entry->usecSince1970 > endTime)
                break;

            /* Got an entry. Colwert it to a sample */
            st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            if(st)
            {
                *Msamples = 0;
                dcgm_mutex_unlock(m_mutex);
                return st;
            }

            (*Msamples)++;
        }
    }
    else /* DCGM_ORDER_DESCENDING */
    {
        /* Which entry we start on depends on if a starting timestamp was provided or not */
        if(!endTime)
        {
            entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
        }
        else
        {
            timeseries_entry_t key;

            key.usecSince1970 = endTime;
            entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key,
                                                                KV_LGE_LESSEQUAL, &cursor);
        }

        /* Walk all samples until we fill our buffer, run out of samples, or go past our end timestamp */
        for(; entry && (*Msamples) < maxSamples;
            entry = (timeseries_entry_p)keyedvector_prev(timeseries->keyedVector, &cursor))
        {
            /* Past our time range? */
            if(startTime && entry->usecSince1970 < startTime)
                break;

            /* Got an entry. Colwert it to a sample */
            st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            if(st)
            {
                *Msamples = 0;
                dcgm_mutex_unlock(m_mutex);
                return st;
            }

            (*Msamples)++;
        }
    }

    /* Handle case where no samples are returned because of lwml errors calling the API */
    if(!(*Msamples))
    {
        if(keyedvector_size(timeseries->keyedVector) > 0)
            retSt = DCGM_ST_NO_DATA; /* User just asked for a time range that has no records */
        else if(watchInfo->lastStatus != LWML_SUCCESS)
            retSt = LwmlReturnToDcgmReturn(watchInfo->lastStatus);
        else if(!watchInfo->isWatched)
            retSt = DCGM_ST_NOT_WATCHED;
        else
            retSt = DCGM_ST_NO_DATA;
    }


    dcgm_mutex_unlock(m_mutex);
    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short dcgmFieldId,
                                               dcgmcm_sample_p sample, DcgmFvBuffer *fvBuffer)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t st, retSt = DCGM_ST_OK;
    timeseries_p timeseries = 0;
    dcgmcm_watch_info_p watchInfo = 0;
    dcgmMutexReturn_t mutexSt;

    if(!sample && !fvBuffer)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
    {
        if(fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, DCGM_ST_UNKNOWN_FIELD);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    /* Handle mutex already being locked by our parent */
    mutexSt = dcgm_mutex_lock_me(m_mutex);

    if(fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId for global field");
        entityGroupId = DCGM_FE_NONE;
    }

    /* Don't need to GetIsValidEntityId(entityGroupId, entityId) here because Get*WatchInfo will
       return null if there isn't a valid watch */

    if(entityGroupId == DCGM_FE_NONE)
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    else
        watchInfo = GetEntityWatchInfo(entityGroupId, entityId, fieldMeta->fieldId, 0);

    st = PrecheckWatchInfoForSamples(watchInfo);
    if(st != DCGM_ST_OK)
    {
        if(mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
            dcgm_mutex_unlock(m_mutex);
        if(fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, st);
        return st;
    }

    /* Data type is assumed to be a time series type */

    timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
    if(!entry)
    {
        /* No entries in time series. If LWML apis failed, return their error code */
        if(mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
            dcgm_mutex_unlock(m_mutex);

        if(watchInfo->lastStatus != LWML_SUCCESS)
            retSt = LwmlReturnToDcgmReturn(watchInfo->lastStatus);
        else if(!watchInfo->isWatched)
            retSt = DCGM_ST_NOT_WATCHED;
        else
            retSt = DCGM_ST_NO_DATA;

        if(fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, retSt);
        return retSt;
    }

    /* Got an entry. Colwert it to a sample */
    if(sample)
    {
        st = DcgmcmTimeSeriesEntryToSample(sample, entry, timeseries);
        retSt = st;
    }
    /* If the user provided a FV buffer, append our sample to it */
    if(fvBuffer)
    {
        st = DcgmcmWriteTimeSeriesEntryToFvBuffer(entityGroupId, entityId, dcgmFieldId, 
                                                  entry, fvBuffer, timeseries);
        retSt = st;
    }

    if(mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetMultipleLatestSamples(std::vector<dcgmGroupEntityPair_t> & entities, 
                                                        std::vector<unsigned short> & fieldIds,
                                                        DcgmFvBuffer *fvBuffer)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;
    std::vector<unsigned short>::iterator fieldIdIt;
    
    if(!fvBuffer)
        return DCGM_ST_BADPARAM;
    
    /* Lock the cache manager once for the whole request */
    dcgm_mutex_lock(m_mutex);

    for(entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        for(fieldIdIt = fieldIds.begin(); fieldIdIt != fieldIds.end(); ++fieldIdIt)
        {
            /* Buffer each sample. Errors are written as statuses for each fv in fvBuffer */
            GetLatestSample((*entityIt).entityGroupId, (*entityIt).entityId,
                            (*fieldIdIt), 0, fvBuffer);
        }
    }

    dcgm_mutex_unlock(m_mutex);
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetValue(int gpuId, unsigned short dcgmFieldId, dcgmcm_sample_p value)
{
    dcgm_field_meta_p fieldMeta = 0;
    lwmlReturn_t lwmlReturn;
    lwmlDevice_t lwmlDevice = 0;    
    timelib64_t now, expireTime;
    dcgmcm_watch_info_p watchInfo = 0;
    dcgmcm_update_thread_t updateCtx;

    if(!value)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    memset(&updateCtx, 0, sizeof(updateCtx));
    ClearThreadCtx(&updateCtx);
    updateCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    updateCtx.entityKey.entityId = gpuId;
    updateCtx.entityKey.fieldId = dcgmFieldId;
    
    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, GpuIdToLwmlIndex(gpuId), dcgmFieldId, 1);
    } else {
        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    }
    
    now = timelib_usecSince1970();
    
    expireTime = 0;
    if(watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;
    
    /* Is the field watched? If so, cause live updates to occur */
    if(watchInfo->isWatched)
        updateCtx.watchInfo = watchInfo;

    /* Do we need a device handle? */
    if(fieldMeta->scope == DCGM_FS_DEVICE)
    {
        lwmlReturn = lwmlDeviceGetHandleByIndex(GpuIdToLwmlIndex(gpuId), &lwmlDevice);
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %u", "lwmlDeviceGetHandleByIndex returned %d for gpuId %u",
                        (int)lwmlReturn, GpuIdToLwmlIndex(gpuId));
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }
    }
    
    switch(fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_AUTOBOOST:
        {
            lwmlEnableState_t enabledState;

            enabledState = value->val.i64 ? LWML_FEATURE_ENABLED : LWML_FEATURE_DISABLED;
            
            lwmlReturn = lwmlDeviceSetDefaultAutoBoostedClocksEnabled(lwmlDevice, enabledState, 0);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }            
            
            if(watchInfo->isWatched)
            {
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now,
                                  expireTime);
            }
            break;
        }
        
        case DCGM_FI_DEV_ENFORCED_POWER_LIMIT:  /* Fall through is intentional */
        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            unsigned int lwrrLimit, minLimit, maxLimit, newPowerLimit;

            newPowerLimit = (unsigned int) (value->val.d * 1000); // colwert from W to mW

            lwmlReturn = lwmlDeviceGetPowerManagementLimit(lwmlDevice, &lwrrLimit);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            lwmlReturn = lwmlDeviceGetPowerManagementLimitConstraints(lwmlDevice, &minLimit, &maxLimit);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if (newPowerLimit < minLimit || newPowerLimit > maxLimit)
            {
                PRINT_WARNING("%u %u %u %u", "gpuId %u. Power limit %u is outside of range %u < x < %u",
                              gpuId, newPowerLimit, minLimit, maxLimit);
                return DCGM_ST_BADPARAM;
            }

            lwmlReturn = lwmlDeviceSetPowerManagementLimit(lwmlDevice, newPowerLimit);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            
            
            if(watchInfo->isWatched)
            {
                AppendEntityDouble(&updateCtx, newPowerLimit/1000, 0, now,
                                   expireTime);
            }            
            
            break;
        }
        
        case DCGM_FI_DEV_APP_SM_CLOCK: // Fall-through is intentional
        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            /* Special Handling as two different values are set simultaneously in this case */
            dcgm_field_meta_p fieldMetaSM = 0;
            dcgm_field_meta_p fieldMetaMEM = 0;
            dcgmcm_watch_info_p watchInfoSM = 0;
            dcgmcm_watch_info_p watchInfoMEM = 0;
            

            fieldMetaSM = DcgmFieldGetById(DCGM_FI_DEV_APP_SM_CLOCK);
            if (!fieldMetaSM)
                return DCGM_ST_UNKNOWN_FIELD;

            watchInfoSM = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_APP_SM_CLOCK, 1);

            fieldMetaMEM = DcgmFieldGetById(DCGM_FI_DEV_APP_MEM_CLOCK);
            if (!fieldMetaMEM)
                return DCGM_ST_UNKNOWN_FIELD;

            watchInfoMEM = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, 1);

            /* Both blank means ignore */
            if(DCGM_INT64_IS_BLANK(value->val.i64) && DCGM_INT64_IS_BLANK(value->val2.i64))
            {
                return LwmlReturnToDcgmReturn(LWML_SUCCESS);
            }
            else if ((value->val.i64 == 0 ) && (value->val2.i64 == 0))
            {
                /* Both 0s means reset application clocks */
                lwmlReturn = lwmlDeviceResetApplicationsClocks(lwmlDevice);
                PRINT_DEBUG("%d", "lwmlDeviceResetApplicationsClocks() returned %d", (int)lwmlReturn);
                if (LWML_SUCCESS != lwmlReturn) {
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }
            else
            {
                /* Set Memory clock and Proc clock pair via LWML */
                lwmlReturn = lwmlDeviceSetApplicationsClocks(lwmlDevice, value->val.i64, value->val2.i64);
                PRINT_DEBUG("%lld %lld %d", "lwmlDeviceSetApplicationsClocks(%lld, %lld) returned %d",
                            value->val.i64, value->val2.i64, (int)lwmlReturn);
                if (LWML_SUCCESS != lwmlReturn) {
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }
            
            if (watchInfoMEM->isWatched) {
                updateCtx.watchInfo = watchInfoMEM;
                updateCtx.entityKey = watchInfoMEM->watchKey;
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
            }            
            
            if (watchInfoSM->isWatched) {
                updateCtx.watchInfo = watchInfoSM;
                updateCtx.entityKey = watchInfoSM->watchKey;
                AppendEntityInt64(&updateCtx, value->val2.i64, 0, now, expireTime);
            }

            break;
        }
        
        case DCGM_FI_DEV_ECC_LWRRENT:   // Fall-through is intentional
        case DCGM_FI_DEV_ECC_PENDING:
        {
            lwmlEnableState_t lwmlState;
            lwmlState = (value->val.i64 == true) ? LWML_FEATURE_ENABLED : LWML_FEATURE_DISABLED;
    
            lwmlReturn = lwmlDeviceSetEccMode(lwmlDevice, lwmlState);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
    
            
            if (watchInfo->isWatched)
            {
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
            }
            
            break;
        }
        
        
        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            lwmlComputeMode_t computeMode;
            
            if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_DEFAULT) 
                computeMode = LWML_COMPUTEMODE_DEFAULT;
            else if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_PROHIBITED)
                computeMode = LWML_COMPUTEMODE_PROHIBITED;
            else if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS)
                computeMode = LWML_COMPUTEMODE_EXCLUSIVE_PROCESS;
            else 
                return DCGM_ST_BADPARAM;
            

            lwmlReturn = lwmlDeviceSetComputeMode(lwmlDevice, computeMode);
            if (LWML_SUCCESS != lwmlReturn) {
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }            
            
            if (watchInfo->isWatched)
            {
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
            }            

            break;
        }

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }
    
    
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddSyncBoostGrp(unsigned int *gpuIdList, unsigned int length, unsigned int *syncBoostId)
{
    dcgm_field_meta_p fieldMeta = 0;
    lwmlReturn_t lwmlReturn;
    lwmlDevice_t *pLwmlDevices = 0;
    timelib64_t now, expireTime;
    dcgmcm_watch_info_p watchInfo = 0;
    lwmlSyncBoostGroupList_t syncBoostListLatest;
    unsigned int countGpus = 0;

    dcgmcm_update_thread_t threadCtx;
    InitAndClearThreadCtx(&threadCtx);
    
    pLwmlDevices = new lwmlDevice_t[length];

    for (unsigned int i = 0; i < length; i++)
    {
        int lwmlIndex = GpuIdToLwmlIndex(gpuIdList[i]);
        if (lwmlIndex < 0)
        {
            PRINT_ERROR("%d %d", "GPU ID to LWML Index colwersion failed for GPU ID %d. Returned: %d",
                        gpuIdList[i], (int)lwmlIndex);
            continue;
        }
        
        lwmlReturn = lwmlDeviceGetHandleByIndex((unsigned int)lwmlIndex, &pLwmlDevices[i]);
        if (lwmlReturn != LWML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "lwmlDeviceGetHandleByIndex returned %d for lwmlIndex %d",
                        (int)lwmlReturn, (int)lwmlIndex);
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }
        
        countGpus++;
    }
    
    if (0 == countGpus) {
        PRINT_ERROR("", "Failed to find any GPUs to add to the sync group");
        delete [] pLwmlDevices;
        return DCGM_ST_BADPARAM;
    }

    fieldMeta = DcgmFieldGetById(DCGM_FI_SYNC_BOOST);
    if (!fieldMeta) {
        delete [] pLwmlDevices;
        return DCGM_ST_UNKNOWN_FIELD;
    }

    watchInfo = GetGlobalWatchInfo(DCGM_FI_SYNC_BOOST, 1);

    threadCtx.entityKey.entityGroupId = DCGM_FE_NONE;
    threadCtx.entityKey.entityId = 0; /* Global fields have no entity ID */
    threadCtx.entityKey.fieldId = fieldMeta->fieldId;
    threadCtx.watchInfo = watchInfo;

    now = timelib_usecSince1970();

    expireTime = 0;
    if(watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;
    
    lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                SystemAddSyncBoostGroup,
                                (pLwmlDevices, countGpus, syncBoostId));
    if(lwmlReturn != LWML_SUCCESS)
    {
        delete [] pLwmlDevices;
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }            
            
    lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
            SystemGetSyncBoostGroups,
            (&syncBoostListLatest));
    watchInfo->lastStatus = lwmlReturn;

    if (lwmlReturn != LWML_SUCCESS) {
        /* Zero out the structure. We're still going to insert it */
        memset(&syncBoostListLatest, 0, sizeof (syncBoostListLatest));
    }

    AppendSyncBoostGroups(&threadCtx, &syncBoostListLatest, now, expireTime);
    
    delete [] pLwmlDevices;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveSyncBoostGrp(unsigned int syncBoostId)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_watch_info_p watchInfo = 0;
    timelib64_t now, expireTime;    
    lwmlReturn_t lwmlReturn;
    lwmlSyncBoostGroupList_t syncBoostListLatest;

    dcgmcm_update_thread_t threadCtx;
    InitAndClearThreadCtx(&threadCtx);
    
    fieldMeta = DcgmFieldGetById(DCGM_FI_SYNC_BOOST);
    if (!fieldMeta) {
        return DCGM_ST_UNKNOWN_FIELD;
    }

    watchInfo = GetGlobalWatchInfo(DCGM_FI_SYNC_BOOST, 1);

    threadCtx.entityKey.entityGroupId = DCGM_FE_NONE;
    threadCtx.entityKey.entityId = 0; /* Global fields have no entity ID */
    threadCtx.entityKey.fieldId = fieldMeta->fieldId;
    threadCtx.watchInfo = watchInfo;

    now = timelib_usecSince1970();

    expireTime = 0;
    if(watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;
    
    lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                SystemRemoveSyncBoostGroup,
                                (syncBoostId));
    if(lwmlReturn != LWML_SUCCESS)
    {
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }            
            
    lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
            SystemGetSyncBoostGroups,
            (&syncBoostListLatest));
    watchInfo->lastStatus = lwmlReturn;

    if (lwmlReturn != LWML_SUCCESS) {
        /* Zero out the structure. We're still going to insert it */
        memset(&syncBoostListLatest, 0, sizeof (syncBoostListLatest));
    }

    AppendSyncBoostGroups(&threadCtx, &syncBoostListLatest, now, expireTime);
    return DCGM_ST_OK;    
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendSamples(DcgmFvBuffer *fvBuffer)
{
    
    if(!fvBuffer)
        return DCGM_ST_BADPARAM;

    dcgmcm_update_thread_t threadCtx;
    InitAndClearThreadCtx(&threadCtx);
    
    /* Lock the mutex for every FV so we only take it once */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock(m_mutex);

    timelib64_t now = timelib_usecSince1970();
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvLwrsor_t cursor = 0;
    
    for(fv = fvBuffer->GetNextFv(&cursor); fv; fv = fvBuffer->GetNextFv(&cursor))
    {
        dcgm_field_meta_t *fieldMeta = DcgmFieldGetById(fv->fieldId);
        if(!fieldMeta)
        {
            PRINT_ERROR("%u", "Unknown fieldId %u in fvBuffer", fv->fieldId);
            continue;
        }

        dcgmcm_watch_info_t *watchInfo;
        if (fieldMeta->scope == DCGM_FS_GLOBAL)
            watchInfo = GetGlobalWatchInfo(fv->fieldId, 1);
        else
        {
            watchInfo = GetEntityWatchInfo((dcgm_field_entity_group_t)fv->entityGroupId, 
                                           fv->entityId, fv->fieldId, 1);
        }

        timelib64_t expireTime = 0;
        if(watchInfo->maxAgeUsec)
            expireTime = now - watchInfo->maxAgeUsec;
        
        threadCtx.watchInfo = watchInfo;
        threadCtx.entityKey = watchInfo->watchKey;

        switch(fv->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(&threadCtx, fv->value.dbl, 0.0, fv->timestamp, expireTime);
                break;
            
            case DCGM_FT_INT64:
                AppendEntityInt64(&threadCtx, fv->value.i64, 0, fv->timestamp, expireTime);
                break;
            
            case DCGM_FT_STRING:
                AppendEntityString(&threadCtx, fv->value.str, fv->timestamp, expireTime);
                break;
            
            case DCGM_FT_BINARY:
            {
                size_t valueSize = (size_t)fv->length - (sizeof(*fv) - sizeof(fv->value));
                AppendEntityBlob(&threadCtx, fv->value.blob, valueSize, fv->timestamp, expireTime);
                break;
            }
            
            default:
                PRINT_ERROR("%u", "Unknown field type: %u", fv->fieldType);
                break;
        }
    }

    if(mutexSt == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendSamples(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                             dcgmcm_sample_p samples, int Nsamples)
{
    return InjectSamples( entityGroupId, entityId, dcgmFieldId, samples, Nsamples );
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::InjectSamples(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                             dcgmcm_sample_p samples, int Nsamples)
{
    dcgm_field_meta_p fieldMeta = 0;
    int sampleIndex;
    dcgmcm_sample_p lwrrentSample = 0;
    dcgmcm_watch_info_p watchInfo = 0;
    timelib64_t now, expireTime;
    dcgmReturn_t retVal = DCGM_ST_OK;

    dcgmcm_update_thread_t threadCtx;
    InitAndClearThreadCtx(&threadCtx);
    threadCtx.fvBuffer = 0;

    if(!dcgmFieldId || !samples || Nsamples < 1)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
        return DCGM_ST_GENERIC_ERROR;
    
    if (fieldMeta->scope == DCGM_FS_GLOBAL)
    {
        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    }
    else
    {
        watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 1);
    }

    if(!watchInfo)
    {
        PRINT_DEBUG("%u %u %u", "InjectSamples eg %u, eid %u, fieldId %u got NULL",
                    entityGroupId, entityId, dcgmFieldId);
        return DCGM_ST_MEMORY;
    }

    /* If anyone is watching this watchInfo, we need to create a 
       fv buffer for the resulting notifcations */
    if(watchInfo->hasSubscribedWatchers)
        threadCtx.fvBuffer = new DcgmFvBuffer();

    threadCtx.watchInfo = watchInfo;
    threadCtx.entityKey.entityGroupId = entityGroupId;
    threadCtx.entityKey.entityId = entityId;
    threadCtx.entityKey.fieldId = fieldMeta->fieldId;
    
    now = timelib_usecSince1970();
    expireTime = 0;
    if(watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;
    /* Update the last queried timestamp to now so that a watch on this field doesn't
       update every single cycle. After all, we are injecting values. If the injected value
       is in the future, then this field won't update live until after the injected timestamp
       Note that we are also updating lastQueriedUsec in the loop below to achieve this. */
    watchInfo->lastQueriedUsec = now;

    /* A future optimization would be to Lock() + Unlock() around this loop
     * and inject all samples at once This is fine for now
     */
    for(sampleIndex = 0; sampleIndex < Nsamples; sampleIndex++)
    {
        lwrrentSample = &samples[sampleIndex];

        /* Use the latest timestamp of the injected samples as the last queried time */
        watchInfo->lastQueriedUsec = DCGM_MAX(now, lwrrentSample->timestamp);

        switch(fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(&threadCtx, lwrrentSample->val.d,
                                   lwrrentSample->val2.d, lwrrentSample->timestamp, expireTime);
                break;

            case DCGM_FT_INT64:
                AppendEntityInt64(&threadCtx, lwrrentSample->val.i64,
                                  lwrrentSample->val2.i64, lwrrentSample->timestamp, expireTime);
                break;

            case DCGM_FT_STRING:
                if(!lwrrentSample->val.str)
                {
                    PRINT_ERROR("%d", "InjectSamples: Null string at index %d of samples",
                                sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityString(&threadCtx, lwrrentSample->val.str,
                                    lwrrentSample->timestamp, expireTime);
                break;

            case DCGM_FT_BINARY:
                if(!lwrrentSample->val.blob)
                {
                    PRINT_ERROR("%d", "InjectSamples: Null blob at index %d of samples",
                                sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityBlob(&threadCtx, lwrrentSample->val.blob,
                                 lwrrentSample->val2.ptrSize,
                                 lwrrentSample->timestamp, expireTime);
                break;

            default:
                PRINT_ERROR("%c", "InjectSamples: Unhandled field type: %c",
                            fieldMeta->fieldType);
                return DCGM_ST_BADPARAM;
        }
    }

    /* Broadcast any aclwmulated notifications */
    if(threadCtx.fvBuffer && threadCtx.affectedSubscribers)
        UpdateFvSubscribers(&threadCtx);

    FreeThreadCtx(&threadCtx);

    return retVal;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::FreeSamples(dcgmcm_sample_p samples, int Nsamples,
                                           unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = 0;
    int sampleIndex;
    dcgmcm_sample_p lwrrentSample = 0;

    if(!dcgmFieldId || !samples || Nsamples < 1)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
        return DCGM_ST_GENERIC_ERROR;

    /* Only strings/binary need their values freed. */
    if ((DCGM_FT_STRING != fieldMeta->fieldType) && (DCGM_FT_BINARY != fieldMeta->fieldType)) {
        return DCGM_ST_OK;
    }

    for(sampleIndex = 0; sampleIndex < Nsamples; sampleIndex++)
    {
        lwrrentSample = &samples[sampleIndex];

        if (fieldMeta->fieldType == DCGM_FT_STRING) {
            if(lwrrentSample->val.str) {
                free(lwrrentSample->val.str);
                lwrrentSample->val.str = 0;
                lwrrentSample->val2.ptrSize = 0;
            }
        }

        if (fieldMeta->fieldType == DCGM_FT_BINARY) {
            if (lwrrentSample->val.blob) {
                free(lwrrentSample->val.blob);
                lwrrentSample->val.blob = 0;
                lwrrentSample->val2.ptrSize = 0;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::FreeThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if(!threadCtx)
        return;

    ClearThreadCtx(threadCtx);

    if(threadCtx->fvBuffer)
        delete threadCtx->fvBuffer;
}

/*****************************************************************************/
void DcgmCacheManager::InitAndClearThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if(!threadCtx)
        return;
    
    threadCtx->fvBuffer = NULL;

    ClearThreadCtx(threadCtx);
}

/*****************************************************************************/
void DcgmCacheManager::ClearThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if(!threadCtx)
        return;
    
    /* Clear the field-values counts */
    memset(threadCtx->numFieldValues, 0, sizeof(threadCtx->numFieldValues));

    threadCtx->watchInfo = 0;
    if(threadCtx->fvBuffer)
        threadCtx->fvBuffer->Clear();
    threadCtx->affectedSubscribers = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFvSubscribers(dcgmcm_update_thread_t *updateCtx)
{
    int numWatcherTypes = 0;
    unsigned int i;
    DcgmWatcherType_t watchers[DcgmWatcherTypeCount];
    
    if(!updateCtx->fvBuffer || !updateCtx->affectedSubscribers)
        return DCGM_ST_OK; /* Nothing to do */

    /* Ok. We've got FVs and subscribers. Let's build the list */
    for(i = 0; i < DcgmWatcherTypeCount; i++)
    {
        if(updateCtx->affectedSubscribers & (1 << i))
        {
            watchers[numWatcherTypes] = (DcgmWatcherType_t)i;
            numWatcherTypes++;
        }
    }
    
    /* Locking the cache manager for now to protect m_onFvUpdateCBs. 
       We can reevaluate this later if there are deadlock issues. Technically,
       we only modify this structure on start-up when we're single threaded. */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);
    
    std::vector<dcgmSubscribedFvUpdateCBEntry_t>::iterator it;

    for(it = m_onFvUpdateCBs.begin(); it != m_onFvUpdateCBs.end(); ++it)
    {
        it->callback(updateCtx->fvBuffer, watchers, numWatcherTypes, it->userData);
    }

    if(mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SubscribeForFvUpdates(dcgmOnSubscribedFvUpdate_f updateCB, void *userData)
{
    dcgmSubscribedFvUpdateCBEntry_t insertEntry;

    if(!updateCB)
        return DCGM_ST_BADPARAM;
    
    insertEntry.callback = updateCB;
    insertEntry.userData = userData;

    dcgm_mutex_lock(m_mutex);
    m_onFvUpdateCBs.push_back(insertEntry);
    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::MarkEnteredDriver()
{
    DcgmLockGuard dlg(m_mutex);

    // Make sure we aren't waiting to detach from the GPUs
    WaitForDriverToBeReady();

    m_inDriverCount++;
}

/*****************************************************************************/
void DcgmCacheManager::MarkReturnedFromDriver()
{
    DcgmLockGuard dlg(m_mutex);

    m_inDriverCount--;
}

/*****************************************************************************/
bool DcgmCacheManager::IsModulePushedFieldId(unsigned int fieldId)
{
    /* LwLink and Profiling fields are the fields >= 700 */
    if(fieldId >= DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00)
        return true;
    else
        return false;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateAllFields(dcgmcm_update_thread_t *threadCtx, 
                                                       timelib64_t *earliestNextUpdate)
{
    int i;
    dcgmReturn_t st;
    dcgmcm_watch_info_p watchInfo = 0;
    void *hashIter;
    timelib64_t now, newNow, age, nextUpdate;
    dcgmMutexReturn_t mutexReturn; /* Tracks the state of the cache manager mutex */
    int anyFieldValues = 0; /* Have we queued any field values to be fetched from lwml? */
    dcgm_field_meta_p fieldMeta = 0;

    mutexReturn = m_mutex->Poll();
    if(mutexReturn != DCGM_MUTEX_ST_LOCKEDBYME)
    {
        PRINT_ERROR("%d", "Entered ActuallyUpdateAllFields() without the lock st %d", (int)mutexReturn);
        return DCGM_ST_GENERIC_ERROR; /* We need the lock in here */
    }

    ClearThreadCtx(threadCtx);

    *earliestNextUpdate = 0;
    now = timelib_usecSince1970();

    /* Walk the hash table of watch objects, looking for any that have expired */
    for(void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter; 
        hashIter = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        
        if(!watchInfo->isWatched)
            continue; /* Not watched */

        /* Some fields are pushed by modules. Don't handle those fields here */
        if(IsModulePushedFieldId(watchInfo->watchKey.fieldId))
            continue;

        /* Last sample time old enough to take another? */
        age = now - watchInfo->lastQueriedUsec;
        if(age < watchInfo->monitorFrequencyUsec)
        {
            nextUpdate = watchInfo->lastQueriedUsec + watchInfo->monitorFrequencyUsec;
            if(!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
            {
                *earliestNextUpdate = nextUpdate;
            }
            continue; /* Not old enough to update */
        }

        fieldMeta = DcgmFieldGetById(watchInfo->watchKey.fieldId);
        if(!fieldMeta)
        {
            PRINT_ERROR("%d", "Unexpected null fieldMeta for field %d", 
                        watchInfo->watchKey.fieldId);
            continue;
        }

        PRINT_DEBUG("%p %u %u %u", "Preparing to update watchInfo %p, eg %u, eid %u, fieldId %u", 
                    watchInfo, watchInfo->watchKey.entityGroupId, watchInfo->watchKey.entityId,
                    watchInfo->watchKey.fieldId);

        if (watchInfo->watchKey.entityGroupId == DCGM_FE_GPU)
        {
            /* Don't cache GPU fields if the GPU is not available */
            DcgmcmGpuStatus_t gpuStatus = GetGpuStatus(watchInfo->watchKey.entityId);
            if(gpuStatus != DcgmcmGpuStatusOk)
            {
                PRINT_DEBUG("%d %d", "Skipping gpuId %d in status %d", 
                            watchInfo->watchKey.entityId, gpuStatus);
                continue;
            }
        }
        /* Base when we sync again on before the driver call so we don't continuously
         * get behind by how long the driver call took
         */
        nextUpdate = now + watchInfo->monitorFrequencyUsec;
        if(!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
        {
            *earliestNextUpdate = nextUpdate;
        }

        /* Set key information before we call child functions */
        threadCtx->entityKey = watchInfo->watchKey;
        threadCtx->watchInfo = watchInfo;

        MarkEnteredDriver();

        /* Unlock the mutex before the driver call, unless we're just buffering a list of field values */
        mutexReturn = m_mutex->Poll();
        if((watchInfo->watchKey.entityGroupId != DCGM_FE_GPU || !fieldMeta->lwmlFieldId) && 
            mutexReturn == DCGM_MUTEX_ST_LOCKEDBYME)
        {
            dcgm_mutex_unlock(m_mutex);
            mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
        }

        if(watchInfo->watchKey.entityGroupId == DCGM_FE_NONE)
            st = BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        else if(watchInfo->watchKey.entityGroupId == DCGM_FE_GPU)
        {
            /* Is this a mapped field? Set aside the info for the field and handle it below */
            if(fieldMeta->lwmlFieldId > 0)
            {
                unsigned int gpuId = watchInfo->watchKey.entityId;
                threadCtx->fieldValueFields[gpuId][threadCtx->numFieldValues[gpuId]] = fieldMeta;
                threadCtx->fieldValueWatchInfo[gpuId][threadCtx->numFieldValues[gpuId]] = watchInfo;
                threadCtx->numFieldValues[gpuId]++;
                anyFieldValues = 1;
                MarkReturnedFromDriver();
                continue;
            }

            st = BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        }
        else if(watchInfo->watchKey.entityGroupId == DCGM_FE_VGPU)
            st = BufferOrCacheLatestVgpuValue(threadCtx, watchInfo->watchKey.entityId, fieldMeta);
        else
            PRINT_DEBUG("%u", "Unhandled entityGroupId %u", watchInfo->watchKey.entityGroupId);
        /* Resync clock after a value fetch since a driver call may take a while */
        newNow = timelib_usecSince1970();

        // accumulate the time spent retrieving this field
        watchInfo->execTimeUsec += newNow - now;
        watchInfo->fetchCount += 1;
        now = newNow;

        /* Relock the mutex if we need to */
        if(mutexReturn == DCGM_MUTEX_ST_NOTLOCKED)
            mutexReturn = dcgm_mutex_lock(m_mutex);

        MarkReturnedFromDriver();
    }

    if(!anyFieldValues)
        return DCGM_ST_OK;

    /* Unlock the mutex before the driver call */
    mutexReturn = m_mutex->Poll();
    if(mutexReturn == DCGM_MUTEX_ST_LOCKEDBYME)
    {
        dcgm_mutex_unlock(m_mutex);
        mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
    }

    for (unsigned int gpuId = 0; gpuId < m_numGpus; gpuId++)
    {
        if(!threadCtx->numFieldValues[gpuId])
            continue;

        PRINT_DEBUG("%d %u", "Got %d field value fields for gpuId %u", 
                    threadCtx->numFieldValues[gpuId], gpuId);
        
        MarkEnteredDriver();
        ActuallyUpdateGpuFieldValues(threadCtx, gpuId);
        MarkReturnedFromDriver();
    }

    /* relock the mutex if we need to */
    if(mutexReturn == DCGM_MUTEX_ST_NOTLOCKED)
        mutexReturn = dcgm_mutex_lock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
static bool FieldSupportsLiveUpdates(dcgm_field_entity_group_t entityGroupId, 
                                     unsigned short fieldId)
{
    if(entityGroupId != DCGM_FE_NONE && entityGroupId != DCGM_FE_GPU)
        return false;
    
    /* Any fieldIds that result in multiple samples need to be excluded from live updates */
    switch(fieldId)
    {
        case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
        case DCGM_FI_DEV_GRAPHICS_PIDS:
        case DCGM_FI_DEV_COMPUTE_PIDS:
        case DCGM_FI_DEV_GPU_UTIL_SAMPLES:
        case DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES:
            return false;

        default:
            break;
    }

    return true;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> & entities, 
                                                            std::vector<unsigned short> & fieldIds,
                                                            DcgmFvBuffer *fvBuffer)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;
    std::vector<unsigned short>::iterator fieldIdIt;
    unsigned short fieldId;
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t dcgmReturn;
    
    if(!fvBuffer)
        return DCGM_ST_BADPARAM;

    /* Allocate a thread context in this function in case we're in a user thread (embeded host engine) */

    dcgmcm_update_thread_t threadCtx;
    InitAndClearThreadCtx(&threadCtx);

    threadCtx.fvBuffer = fvBuffer;

    /* Note: because we're handling fields that come from the LWML field value APIs out of order
             from those that don't, we don't guarantee any order of returned results */

    for(entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        dcgm_field_entity_group_t entityGroupId = (*entityIt).entityGroupId;
        dcgm_field_eid_t entityId = (*entityIt).entityId;

        threadCtx.entityKey.entityGroupId = entityGroupId;
        threadCtx.entityKey.entityId = entityId;

        for(fieldIdIt = fieldIds.begin(); fieldIdIt != fieldIds.end(); ++fieldIdIt)
        {
            fieldId = *fieldIdIt;
            threadCtx.entityKey.fieldId = fieldId;
            /* Restore entityGroupId since it may have been changed by a previous iteration */
            entityGroupId = (*entityIt).entityGroupId;

            fieldMeta = DcgmFieldGetById(fieldId);
            if(!fieldMeta)
            {
                PRINT_ERROR("%u", "Invalid field ID %u passed in.", fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, 
                                        DCGM_ST_UNKNOWN_FIELD);
                continue;
            }
            
            /* Handle if the user accidentally marked this field as an entity field if it's global */
            if(fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                PRINT_DEBUG("%u", "Fixed entityGroupId to be DCGM_FE_NONE fieldId %u", fieldId);
                entityGroupId = DCGM_FE_NONE;
            }

            /* Does this entity + fieldId even support live updates? 
               This will filter out VGPUs and LwSwitches */
            if(!FieldSupportsLiveUpdates(entityGroupId, fieldId))
            {
                PRINT_DEBUG("%u %u", "eg %u fieldId %u doesn't support live updates.", 
                            entityGroupId, fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, 
                                        DCGM_ST_FIELD_UNSUPPORTED_BY_API);
                continue;
            }

            if(entityGroupId == DCGM_FE_NONE)
            {
                dcgmReturn = BufferOrCacheLatestGpuValue(&threadCtx, fieldMeta);
            }
            else if(entityGroupId == DCGM_FE_GPU)
            {
                /* Is the entityId valid? */
                if(!GetIsValidEntityId(entityGroupId, entityId))
                {
                    PRINT_WARNING("%u %u", "Got invalid eg %u, eid %u", entityGroupId, entityId);
                    fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_BADPARAM);
                    continue;
                }

                /* Is this a mapped field? Set aside the info for the field and handle it below */
                if(fieldMeta->lwmlFieldId > 0)
                {
                    threadCtx.fieldValueFields[entityId][threadCtx.numFieldValues[entityId]] = fieldMeta;
                    threadCtx.fieldValueWatchInfo[entityId][threadCtx.numFieldValues[entityId]] = 0; /* Don't cache. Only buffer it */
                    threadCtx.numFieldValues[entityId]++;
                }
                else
                    dcgmReturn = BufferOrCacheLatestGpuValue(&threadCtx, fieldMeta);
            }
            else
            {
                /* Unhandled entity group. Should have been caught by FieldSupportsLiveUpdates() */
                PRINT_ERROR("%u %u %u", "Didn't expect to get here for eg %u, eid %u, fieldId %u", 
                            threadCtx.entityKey.entityGroupId,threadCtx.entityKey.entityId, fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_FIELD_UNSUPPORTED_BY_API);
            }
        }

        /* Handle any field values that come from the LWML FV APIs. Note that entityId could be invalid, so
           we need to check it */
        if(entityGroupId == DCGM_FE_GPU && 
           GetIsValidEntityId(entityGroupId, entityId) &&
           threadCtx.numFieldValues[entityId] > 0)
        {
            ActuallyUpdateGpuFieldValues(&threadCtx, entityId);
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static double LwmlFieldValueToDouble(lwmlFieldValue_t *v)
{
    long long retVal = 0;

    switch(v->valueType)
    {
        case LWML_VALUE_TYPE_DOUBLE:
            return (double)v->value.dVal;

        case LWML_VALUE_TYPE_UNSIGNED_INT:
            return (double)v->value.uiVal;

        case LWML_VALUE_TYPE_UNSIGNED_LONG:
            return (double)v->value.ulVal;

        case LWML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            return (double)v->value.ullVal;

        case LWML_VALUE_TYPE_SIGNED_LONG_LONG:
            return (double)v->value.sllVal;

        default:
            PRINT_ERROR("%d", "Unhandled valueType: %d", (int)v->valueType);
            return retVal;
    }

    return retVal;
}

/*****************************************************************************/
long long LwmlFieldValueToInt64(lwmlFieldValue_t *v)
{
    long long retVal = 0;

    switch(v->valueType)
    {
        case LWML_VALUE_TYPE_DOUBLE:
            return (long long)v->value.dVal;

        case LWML_VALUE_TYPE_UNSIGNED_INT:
            return (long long)v->value.uiVal;

        case LWML_VALUE_TYPE_UNSIGNED_LONG:
            return (long long)v->value.ulVal;

        case LWML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            return (long long)v->value.ullVal;

        case LWML_VALUE_TYPE_SIGNED_LONG_LONG:
            return (long long)v->value.sllVal;

        default:
            PRINT_ERROR("%d", "Unhandled valueType: %d", (int)v->valueType);
            return retVal;
    }

    return retVal;
}

void DcgmCacheManager::InsertLwmlErrorValue(dcgmcm_update_thread_t *threadCtx, unsigned char fieldType,
                                            lwmlReturn_t err, timelib64_t maxAgeUsec)

{
    timelib64_t now = timelib_usecSince1970();
    timelib64_t oldestKeepTime = 0;

    if (maxAgeUsec)
        oldestKeepTime = now - maxAgeUsec;

    /* Append a blank value of the correct type for our fieldId */
    switch (fieldType)
    {
        case DCGM_FT_DOUBLE:
            AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(err), 0, now, oldestKeepTime);
            break;

        case DCGM_FT_INT64:
            AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(err), 0, now, oldestKeepTime);
            break;

        case DCGM_FT_STRING:
            AppendEntityString(threadCtx, LwmlErrorToStringValue(err), now, oldestKeepTime);
            break;

        default:
            PRINT_ERROR("%c", "Field Type %c is unsupported for colwersion from LWML errors", fieldType);
            break;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateGpuFieldValues(dcgmcm_update_thread_t *threadCtx, 
                                                            unsigned int gpuId)
{
    lwmlFieldValue_t values[LWML_FI_MAX]; /* Place to store actual LWML field values */
    lwmlFieldValue_t *fv;                 /* Cached field value pointer */
    int i;
    lwmlReturn_t lwmlReturn;
    timelib64_t expireTime;

    /* Make local variables for threadCtx members to simplify the code */
    int numFields = threadCtx->numFieldValues[gpuId];
    dcgm_field_meta_p *fieldMeta = threadCtx->fieldValueFields[gpuId];
    dcgmcm_watch_info_p *watchInfo = threadCtx->fieldValueWatchInfo[gpuId];

    if (gpuId >= m_numGpus)
        return DCGM_ST_GENERIC_ERROR;

    if(numFields >= LWML_FI_MAX)
    {
        PRINT_CRITICAL("%d", "numFieldValueFields %d > LWML_FI_MAX", numFields);
        return DCGM_ST_BADPARAM;
    }

    /* Initialize the values[] array */
    memset(&values[0], 0, sizeof(values[0])*numFields);
    for(i = 0; i < numFields; i++)
    {
        values[i].fieldId = fieldMeta[i]->lwmlFieldId;
    }

    // Do not attempt to poll LWML for values of detached GPUs
    if (m_gpus[gpuId].status != DcgmcmGpuStatusDetached)
    {
        /* The fieldId field of fieldValueValues[] was already populated above. Make the LWML call */
        lwmlReturn = lwmlDeviceGetFieldValues(m_gpus[gpuId].lwmlDevice, numFields, &values[0]);
        if(lwmlReturn != LWML_SUCCESS)
        {
            /* Any given field failure will be on a single fieldValueValues[] entry. A global failure is
             * unexpected */
            PRINT_ERROR("%d", "Unexpected LWML return %d from lwmlDeviceGetFieldValues", lwmlReturn);
            return LwmlReturnToDcgmReturn(lwmlReturn);
        }
    }

    /* Set thread context variables that won't change */
    threadCtx->entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx->entityKey.entityId = gpuId;

    for(i = 0; i < numFields; i++)
    {
        fv = &values[i];

        /* Set threadCtx variables before we possibly use them */
        threadCtx->entityKey.fieldId = fieldMeta[i]->fieldId;
        threadCtx->watchInfo = watchInfo[i];

        if (m_gpus[gpuId].status == DcgmcmGpuStatusDetached)
        {
            // Detached GPUs get an intentional LWML error and we're done
            InsertLwmlErrorValue(threadCtx, fieldMeta[i]->fieldType, LWML_ERROR_GPU_IS_LOST,
                                 watchInfo[i]->maxAgeUsec);
            continue;
        }

        /* We need a timestamp on every FV or we'll just keep saving it over and over */
        bool isTimestampReinit = false;
        if(!fv->timestamp)
        {
            PRINT_DEBUG("%u %u %i", "gpuId %u, fieldId %u, index %d had a null timestamp.",
                        gpuId, fv->fieldId, i);
            fv->timestamp = timelib_usecSince1970();
            isTimestampReinit = true;

            /* WaR for LWML bug 2009232 where fields ECC can be left uninitialized if ECC is disabled */

            if(!fv->latencyUsec && fv->valueType == LWML_VALUE_TYPE_DOUBLE /* 0 */ &&
               fv->fieldId >= LWML_FI_DEV_ECC_LWRRENT && fv->fieldId <= LWML_FI_DEV_RETIRED_PENDING)
            {
                if(fv->fieldId > LWML_FI_DEV_ECC_PENDING)
                    fv->lwmlReturn = LWML_ERROR_NOT_SUPPORTED;
                else
                {
                    /* Read current/pending manually so that they have a valid value */
                    lwmlEnableState_t lwrrentIsEnabled, pendingIsEnabled;
                    fv->lwmlReturn = lwmlDeviceGetEccMode(m_gpus[gpuId].lwmlDevice, &lwrrentIsEnabled,
                            &pendingIsEnabled);
                    fv->valueType = LWML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                    if(fv->fieldId == LWML_FI_DEV_ECC_LWRRENT)
                        fv->value.ullVal = lwrrentIsEnabled;
                    else
                        fv->value.ullVal = pendingIsEnabled;
                }
            }
        }

        /* Expiration is either measured in absolute time or 0 */
        expireTime                             = 0;
        timelib64_t const now                  = timelib_usecSince1970();
        timelib64_t previouslyQueriedTimestamp = now;
        if(watchInfo[i])
        {
            if (watchInfo[i]->lastQueriedUsec != 0)
            {
                previouslyQueriedTimestamp = watchInfo[i]->lastQueriedUsec;
            }

            if (watchInfo[i]->maxAgeUsec)
            {
                expireTime = fv->timestamp - watchInfo[i]->maxAgeUsec;
            }
            watchInfo[i]->execTimeUsec += fv->latencyUsec;
            watchInfo[i]->fetchCount++;
            watchInfo[i]->lastQueriedUsec = fv->timestamp;
            watchInfo[i]->lastStatus = fv->lwmlReturn;
        }

        /* WAR for LWML Bug 2032468. Force the valueType to unsigned long long for ECC fields, because LWML
         * isn't setting them and it's defaulting to a double which doesn't get stored properly. */
        if ((threadCtx->entityKey.fieldId <= DCGM_FI_DEV_ECC_DBE_AGG_TEX)
            && (threadCtx->entityKey.fieldId >= DCGM_FI_DEV_ECC_LWRRENT))
        {
            fv->valueType = LWML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        }
        else if (fv->lwmlReturn == LWML_SUCCESS && DCGM_FI_DEV_LWLINK_BANDWIDTH_L0 <= threadCtx->entityKey.fieldId
                 && threadCtx->entityKey.fieldId <= DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL)
        {
            /* LwLink fields that are flits need to be multiplied by 16 per
             * https://confluence.lwpu.com/display/CSSRM/Bug+2441268+-+lwml+lwlink+APIs+return+incorrect+byte+counts
             *
             * We report lwlink bandwidth in MB/s units
             */
            double tmpVal = fv->value.ullVal;

            tmpVal *= double(m_gpus[gpuId].lwLinkCountersAreFlits ? 16 : 1) / 1000000.0; // MB
            if (now != previouslyQueriedTimestamp && !isTimestampReinit)
            {
                /* the very first measurement may have a spike, as we will not devide it by elapsed time */
                tmpVal /= double(now - previouslyQueriedTimestamp) / 1000000.0;
            }

            fv->value.ullVal = llround(tmpVal);

            switch (threadCtx->entityKey.fieldId)
            {
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L0:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 0, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L1:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 1, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L2:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 2, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L3:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 3, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L4:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 4, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_L5:
                    lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                        m_gpus[gpuId].lwmlDevice, 5, DCGMCM_LWLINK_COUNTER_BYTES);
                    break;
                case DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL:
                    for (int link = 0; link < DCGM_LWLINK_MAX_LINKS_PER_GPU; ++link)
                    {
                        if (m_gpus[gpuId].lwLinkLinkState[link] == DcgmLwLinkLinkStateUp)
                        {
                            lwmlReturn = lwmlDeviceResetLwLinkUtilizationCounter(
                                m_gpus[gpuId].lwmlDevice, link, DCGMCM_LWLINK_COUNTER_BYTES);
                            if (LWML_SUCCESS != lwmlReturn)
                            {
                                PRINT_DEBUG("%d %d %d",
                                            "LWLink counter reset failure. GpuId: %d, Link: %d, Error: %d",
                                            gpuId,
                                            link,
                                            (int)lwmlReturn);
                                break;
                            }
                        }
                    }
                    break;
                default:
                    PRINT_ERROR("%d", "Unknown field id: %d", threadCtx->entityKey.fieldId);
                    break;
            }

            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_ERROR("%d", "Unable to reset LWLink counter(s). LWML result: %d", (int)lwmlReturn);
                fv->lwmlReturn = lwmlReturn;
            }
        }

        if(fv->lwmlReturn != LWML_SUCCESS)
        {
            /* Store an appropriate error for the destination type */
            timelib64_t maxAgeUsec = 0;
            if (watchInfo[i])
                maxAgeUsec = watchInfo[i]->maxAgeUsec;
            InsertLwmlErrorValue(threadCtx, fieldMeta[i]->fieldType, fv->lwmlReturn, maxAgeUsec);
        }
        else /* LWML_SUCCESS */
        {
            PRINT_DEBUG("%d", "fieldId %d got good value", fv->fieldId);

            /* Store an appropriate error for the destination type */
            switch(fieldMeta[i]->fieldType)
            {
                case DCGM_FT_INT64:
                    AppendEntityInt64(threadCtx, LwmlFieldValueToInt64(fv), 0,
                                      (timelib64_t)fv->timestamp, expireTime);
                    break;

                case DCGM_FT_DOUBLE:
                    AppendEntityDouble(threadCtx, LwmlFieldValueToDouble(fv), 0.0,
                                       (timelib64_t)fv->timestamp, expireTime);
                    break;

                default:
                    PRINT_ERROR("%c", "Unhandled field value output type: %c", fieldMeta[i]->fieldType);
                    break;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ClearWatchInfo(dcgmcm_watch_info_p watchInfo, int clearCache)
{
    if(!watchInfo)
        return;
    
    watchInfo->watchers.clear();
    watchInfo->isWatched = 0;
    watchInfo->monitorFrequencyUsec = 0;
    watchInfo->maxAgeUsec = 0;
    watchInfo->lastQueriedUsec = 0;
    if(watchInfo->timeSeries && clearCache)
    {
        timeseries_destroy(watchInfo->timeSeries);
        watchInfo->timeSeries = 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ClearAllEntities(int clearCache)
{
    dcgmReturn_t st;
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int nextFieldIndex = 0;
    dcgmMutexReturn_t mutexReturn;
    int numCleared = 0;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Walk the watch table and clear every entry */
    for(void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter; 
        hashIter = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        numCleared++;
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        ClearWatchInfo(watchInfo, clearCache);
    }

    if(mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%d %d", "ClearAllEntities clearCache %d, numCleared %d",
                clearCache, numCleared);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ClearEntity(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            int clearCache)
{
    dcgmReturn_t st;
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int nextFieldIndex = 0;
    dcgmMutexReturn_t mutexReturn;
    int numMatched = 0;
    int numScanned = 0;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Walk the watch table and clear anything that points at this entityGroup + entityId combo */
    for(void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter; 
        hashIter = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        numScanned++;
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);

        if(watchInfo->watchKey.entityGroupId != entityGroupId
           || watchInfo->watchKey.entityId != entityId)
        {
            continue; /* Not a match */
        }

        numMatched++;
        ClearWatchInfo(watchInfo, clearCache);
    }

    if(mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %u %d %d %d", "ClearEntity eg %u, eid %u, clearCache %d, "
                "numScanned %d, numMatched %d",
                entityGroupId, entityId, clearCache, numScanned, numMatched);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::RunLockStep(dcgmcm_update_thread_t *threadCtx)
{
    int st;
    int haveLock = 0;
    unsigned int sleepAtATimeMs = 10000;
    timelib64_t earliestNextUpdate, lastWakeupTime, now;

    lastWakeupTime = 0;

    while(!ShouldStop())
    {
        if(!haveLock)
            dcgm_mutex_lock(m_mutex);
        haveLock = 1;

        /* Update runtime stats */
        m_runStats.numSleepsDone++;
        m_runStats.lockCount = m_mutex->GetLockCount();
        m_runStats.sleepTimeUsec += 1000 * ((long long)sleepAtATimeMs);
        now = timelib_usecSince1970();
        /* If lastWakeupTime != 0 then we actually work up to do work */
        if(lastWakeupTime)
            m_runStats.awakeTimeUsec += now - lastWakeupTime;

#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%u %lld %lld", "Waiting on m_startUpdateCondition for %u ms. updateCycleFinished %lld. was awake for %lld usec",
                    sleepAtATimeMs, m_runStats.updateCycleFinished, (long long)now - lastWakeupTime);
#endif

        /* Wait for someone to call UpdateAllFields(). Check if shouldFinishCycle changed before we got the lock.
         * If so, we need to skip our sleep and do another update cycle */
        if(m_runStats.updateCycleFinished >= m_runStats.shouldFinishCycle)
        {
            st = m_mutex->CondWait(&m_startUpdateCondition, sleepAtATimeMs);
#ifdef DEBUG_UPDATE_LOOP
            PRINT_DEBUG("%d %lld %lld", "Woke up to st %d. updateCycleFinished %lld, shouldFinishCycle %lld",
                        st, m_runStats.updateCycleFinished, m_runStats.shouldFinishCycle);
        }
        else
        {
            PRINT_DEBUG("", "RunLockStep() skipped lwosCondWait()");
#endif
        }


        if(m_runStats.updateCycleFinished >= m_runStats.shouldFinishCycle)
        {
            lastWakeupTime = 0;
            continue; /* Keep lock. We need it for next lwosCondWait */
        }

        lastWakeupTime = timelib_usecSince1970();

        m_runStats.updateCycleStarted++;

        /* Leave the mutex locked throughout the update loop. It will be unlocked before any driver calls */

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if(!threadCtx->fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx->fvBuffer = new DcgmFvBuffer();
            if(!threadCtx->fvBuffer)
                PRINT_ERROR("", "Got NULL fvBuffer");
        }

        /* Try to update all fields */
        earliestNextUpdate = 0;
        ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);

        if(threadCtx->fvBuffer)
            UpdateFvSubscribers(threadCtx);

        m_runStats.updateCycleFinished++;
#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%lld", "Setting m_updateCompleteCondition at updateCycleFinished %lld",
                    m_runStats.updateCycleFinished);
#endif
        /* Let anyone waiting on this update cycle know we're done */
        lwosCondBroadcast(&m_updateCompleteCondition);
        dcgm_mutex_unlock(m_mutex);
        haveLock = 0;
    }

    if(haveLock)
        dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/
void DcgmCacheManager::RunTimedWakeup(dcgmcm_update_thread_t *threadCtx)
{
    timelib64_t now, maxNextWakeTime, diff, sleepFor, earliestNextUpdate, startOfLoop;
    timelib64_t wakeTimeInterval = 10000000;
    unsigned int sleepAtATimeMs = 1000;
    int st;

    while(!ShouldStop())
    {
        startOfLoop = timelib_usecSince1970();
        /* Maximum time of 10 second between loops */
        maxNextWakeTime = startOfLoop + wakeTimeInterval;

        dcgm_mutex_lock(m_mutex);
        m_runStats.updateCycleStarted++;

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if(!threadCtx->fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx->fvBuffer = new DcgmFvBuffer();
            if(!threadCtx->fvBuffer)
                PRINT_ERROR("", "Got NULL fvBuffer");
        }

        /* Try to update all fields */
        earliestNextUpdate = 0;
        ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);

        if(threadCtx->fvBuffer)
            UpdateFvSubscribers(threadCtx);

        m_runStats.updateCycleFinished++;
        /* Let anyone waiting on this update cycle know we're done */
        lwosCondBroadcast(&m_updateCompleteCondition);
        dcgm_mutex_unlock(m_mutex);

        /* Resync */
        now = timelib_usecSince1970();
        m_runStats.awakeTimeUsec += (now - startOfLoop);

        /* Only bother if we are supposed to sleep for > 100 usec. Sleep takes 60+ usec */
        /* Are we past our maximum time between loops? */
        if(now > maxNextWakeTime - 100)
        {
            //printf("No sleep. now %lld. maxNextWakeTime %lld\n",
            //       (long long int)diff, (long long int)maxNextWakeTime);
            m_runStats.numSleepsSkipped++;
            continue;
        }

        /* If we need to update something earlier than max time, sleep until we know
         * we have to update something
         */
        diff = maxNextWakeTime - now;
        if(earliestNextUpdate && earliestNextUpdate < maxNextWakeTime)
            diff = earliestNextUpdate - now;

        if(diff < 1000)
        {
            //printf("No sleep. diff %lld\n", (long long int)diff);
            m_runStats.numSleepsSkipped++;
            continue;
        }

        sleepAtATimeMs = diff / 1000;
        m_runStats.sleepTimeUsec += diff;
        m_runStats.lockCount = m_mutex->GetLockCount();
        m_runStats.numSleepsDone++;
        //printf("Sleeping for %u\n", sleepAtATimeMs);

        /* Sleep for diff usec. This is an interruptible wait in case someone calls UpdateAllFields() */
        dcgm_mutex_lock(m_mutex);
        st = m_mutex->CondWait(&m_startUpdateCondition, sleepAtATimeMs);
        /* We don't care about st. Either it timed out or we were woken up. Either way, run
         * another update loop
         */
        dcgm_mutex_unlock(m_mutex);
    }
}

/*****************************************************************************/
void DcgmCacheManager::run(void)
{
    dcgmcm_update_thread_t *updateThreadCtx;

    updateThreadCtx = (dcgmcm_update_thread_t *)malloc(sizeof(*updateThreadCtx));
    if(!updateThreadCtx)
    {
        PRINT_ERROR("", "Unable to alloc updateThreadCtx. Exiting update thread");
        return;
    }
    memset(updateThreadCtx, 0, sizeof(*updateThreadCtx));

    PRINT_INFO("", "Cache manager update thread starting");

    if(m_pollInLockStep)
        RunLockStep(updateThreadCtx);
    else
        RunTimedWakeup(updateThreadCtx);

    FreeThreadCtx(updateThreadCtx);
    free(updateThreadCtx);

    PRINT_INFO("", "Cache manager update thread ending");
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgm_field_meta_p fieldMeta = 0;
    timeseries_p timeseries = 0;

    if(!fieldInfo)
        return DCGM_ST_BADPARAM;

    if(fieldInfo->version != dcgmCacheManagerFieldInfo_version)
    {
        PRINT_ERROR("%d %d", "Got GetCacheManagerFieldInfo ver %d != expected %d",
                    (int)fieldInfo->version, (int)dcgmCacheManagerFieldInfo_version);
        return DCGM_ST_VER_MISMATCH;
    }

    fieldMeta = DcgmFieldGetById(fieldInfo->fieldId);
    if(!fieldMeta)
    {
        PRINT_ERROR("%u", "Invalid fieldId %u passed to GetCacheManagerFieldInfo",
                    (unsigned int)fieldInfo->fieldId);
        return DCGM_ST_BADPARAM;
    }

    
    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        if(fieldInfo->gpuId >= m_numGpus)
        {
            PRINT_ERROR("%u", "Invalid gpuId %u passed to GetCacheManagerFieldInfo",
                    fieldInfo->gpuId);
            return DCGM_ST_BADPARAM;
        }

        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, fieldInfo->gpuId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }
    if(!watchInfo)
    {
        PRINT_DEBUG("", "not watched.");
        return DCGM_ST_NOT_WATCHED;
    }

    dcgm_mutex_lock(m_mutex);
    /* UNLOCK AFTER HERE */

    /* Populate the fields we can */
    fieldInfo->flags = 0;
    if(watchInfo->isWatched)
        fieldInfo->flags |= DCGM_CMI_F_WATCHED;

    fieldInfo->version = dcgmCacheManagerFieldInfo_version;
    fieldInfo->lastStatus = (short)watchInfo->lastStatus;
    fieldInfo->maxAgeUsec = watchInfo->maxAgeUsec;
    fieldInfo->monitorFrequencyUsec = watchInfo->monitorFrequencyUsec;
    fieldInfo->fetchCount = watchInfo->fetchCount;
    fieldInfo->execTimeUsec = watchInfo->execTimeUsec;

    fieldInfo->numWatchers = 0;
    std::vector<dcgm_watch_watcher_info_t>::iterator it;
    for(it = watchInfo->watchers.begin(); 
        it != watchInfo->watchers.end() && fieldInfo->numWatchers < DCGM_CM_FIELD_INFO_NUM_WATCHERS; 
        ++it)
    {
        dcgm_cm_field_info_watcher_t *watcher = &fieldInfo->watchers[fieldInfo->numWatchers];
        watcher->watcherType = it->watcher.watcherType;
        watcher->connectionId = it->watcher.connectionId;
        watcher->monitorFrequencyUsec = it->monitorFrequencyUsec;
        watcher->maxAgeUsec = it->maxAgeUsec;
        fieldInfo->numWatchers++;
    }

    if(!watchInfo->timeSeries)
    {
        /* No values yet */
        dcgm_mutex_unlock(m_mutex);
        fieldInfo->newestTimestamp = 0;
        fieldInfo->oldestTimestamp = 0;
        fieldInfo->numSamples = 0;
        return DCGM_ST_OK;
    }

    timeseries = watchInfo->timeSeries;
    kv_lwrsor_t cursor;
    timeseries_entry_p entry = 0;

    fieldInfo->numSamples = keyedvector_size(timeseries->keyedVector);
    if(!fieldInfo->numSamples)
    {
        /* No values yet */
        dcgm_mutex_unlock(m_mutex);
        fieldInfo->newestTimestamp = 0;
        fieldInfo->oldestTimestamp = 0;
        return DCGM_ST_OK;
    }

    /* Get the first and last records to get their timestamps */
    entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);
    fieldInfo->oldestTimestamp = entry->usecSince1970;
    entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
    fieldInfo->newestTimestamp = entry->usecSince1970;

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::MarkSubscribersInThreadCtx(dcgmcm_update_thread_t *threadCtx, 
                                                  dcgmcm_watch_info_p watchInfo)
{
    if(!threadCtx || !watchInfo)
        return;
    
    /* Fast path exit if there are no subscribers */
    if(!watchInfo->hasSubscribedWatchers)
        return;
    
    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    for(it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if(!it->isSubscribed)
            continue;

        threadCtx->affectedSubscribers |= 1 << it->watcher.watcherType;

        PRINT_DEBUG("%u %u %u %u", "watcherType %u has a subscribed update to eg %u, eid %u, fieldId %u",
                    it->watcher.watcherType, watchInfo->watchKey.entityGroupId, 
                    watchInfo->watchKey.entityId, watchInfo->watchKey.fieldId);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityDouble(dcgmcm_update_thread_t *threadCtx, 
                                                  double value1, double value2,
                                                  timelib64_t timestamp, 
                                                  timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if(threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddDoubleValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId, 
                                            threadCtx->entityKey.entityId,
                                            threadCtx->entityKey.fieldId, value1, 
                                            timestamp, DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    /* Should we cache the value? */
    if(watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if(!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_DOUBLE);
            if(dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_double_coerce(watchInfo->timeSeries, timestamp, value1, value2);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if(mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    PRINT_DEBUG("%u %u %u %lld %f %f %d %d", "Appended entity double eg %u, eid %u, fieldId %u, ts %lld, value1 %f, value2 %f, cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId, threadCtx->entityKey.entityId, threadCtx->entityKey.fieldId, 
                (long long)timestamp, value1, value2, watchInfo ? 1 : 0, threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AllocWatchInfoTimeSeries(dcgmcm_watch_info_p watchInfo, int tsType)
{
    if(watchInfo->timeSeries)
        return DCGM_ST_OK; /* Already alloc'd */

    int errorSt = 0;   
    watchInfo->timeSeries = timeseries_alloc(tsType, &errorSt);
    if(!watchInfo->timeSeries)
    {
        PRINT_ERROR("%d %d", "timeseries_alloc(tsType=%d) failed with %d", tsType, errorSt);
        return DCGM_ST_MEMORY; /* Assuming it's a memory alloc error */
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::EnforceWatchInfoQuota(dcgmcm_watch_info_p watchInfo, 
                                                     timelib64_t timestamp, 
                                                     timelib64_t oldestKeepTimestamp)
{
    if(!watchInfo || !watchInfo->timeSeries)
        return DCGM_ST_OK; /* Nothing to do */
    
    /* Passing count quota as 0 since we enforce quota by time alone */
    int st = timeseries_enforce_quota(watchInfo->timeSeries, oldestKeepTimestamp, 0);
    if(st)
    {
        PRINT_ERROR("%d", "timeseries_enforce_quota returned %d", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityInt64(dcgmcm_update_thread_t *threadCtx, 
                                                 long long value1, long long value2, 
                                                 timelib64_t timestamp, timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if(threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddInt64Value((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId, 
                                           threadCtx->entityKey.entityId,
                                           threadCtx->entityKey.fieldId, value1, 
                                           timestamp, DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    if(watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if(!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_INT64);
            if(dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_int64_coerce(watchInfo->timeSeries, timestamp, value1, value2);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if(mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    PRINT_DEBUG("%u %u %u %lld %lld %lld %d %d", "Appended entity i64 eg %u, eid %u, fieldId %u, ts %lld, value1 %lld, value2 %lld, cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId, threadCtx->entityKey.entityId,threadCtx->entityKey.fieldId, 
                (long long)timestamp, value1, value2, watchInfo ? 1 : 0, threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityString(dcgmcm_update_thread_t *threadCtx, char *value, 
                                                  timelib64_t timestamp, timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if(threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddStringValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId, 
                                            threadCtx->entityKey.entityId,
                                            threadCtx->entityKey.fieldId, value, 
                                            timestamp, DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }
    
    if(watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if(!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_STRING);
            if(dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_string(watchInfo->timeSeries, timestamp, value);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if(mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    PRINT_DEBUG("%u %u %u %lld %s %d %d", "Appended entity string eg %u, eid %u, fieldId %u, ts %lld, value \"%s\", cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId, threadCtx->entityKey.entityId,threadCtx->entityKey.fieldId, 
                (long long)timestamp, value, watchInfo ? 1 : 0, threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityBlob(dcgmcm_update_thread_t *threadCtx, void *value, int valueSize,
                                                timelib64_t timestamp, timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if(threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddBlobValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId, 
                                            threadCtx->entityKey.entityId,
                                            threadCtx->entityKey.fieldId, value, valueSize,
                                            timestamp, DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    if(watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if(!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_BLOB);
            if(dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_blob(watchInfo->timeSeries, timestamp, value, valueSize);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if(mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    PRINT_DEBUG("%u %u %u %lld %d %d %d", "Appended entity blob eg %u, eid %u, fieldId %u, ts %lld, valueSize %d, cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId, threadCtx->entityKey.entityId, 
                threadCtx->entityKey.fieldId, (long long)timestamp, valueSize,
                watchInfo ? 1 : 0, threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
char *DcgmCacheManager::LwmlErrorToStringValue(lwmlReturn_t lwmlReturn)
{
    switch(lwmlReturn)
    {
        case LWML_SUCCESS:
            LW_ASSERT(0); /* Should never get here */
            break;

        case LWML_ERROR_NOT_SUPPORTED:
            return (char *)DCGM_STR_NOT_SUPPORTED;

        case LWML_ERROR_NO_PERMISSION:
            return (char *)DCGM_STR_NOT_PERMISSIONED;

        case LWML_ERROR_NOT_FOUND:
            return (char *)DCGM_STR_NOT_FOUND;

        default:
            return (char *)DCGM_STR_BLANK;
    }

    return (char *)DCGM_STR_BLANK;
}

/*****************************************************************************/
long long DcgmCacheManager::LwmlErrorToInt64Value(lwmlReturn_t lwmlReturn)
{
    switch(lwmlReturn)
    {
        case LWML_SUCCESS:
            LW_ASSERT(0); /* Should never get here */
            break;

        case LWML_ERROR_NOT_SUPPORTED:
            return DCGM_INT64_NOT_SUPPORTED;

        case LWML_ERROR_NO_PERMISSION:
            return DCGM_INT64_NOT_PERMISSIONED;

        case LWML_ERROR_NOT_FOUND:
            return DCGM_INT64_NOT_FOUND;

        default:
            return DCGM_INT64_BLANK;
    }

    return DCGM_INT64_BLANK;
}

/*****************************************************************************/
int DcgmCacheManager::LwmlErrorToInt32Value(lwmlReturn_t lwmlReturn)
{
    switch(lwmlReturn)
    {
        case LWML_SUCCESS:
            LW_ASSERT(0); /* Should never get here */
            break;

        case LWML_ERROR_NOT_SUPPORTED:
            return DCGM_INT32_NOT_SUPPORTED;

        case LWML_ERROR_NO_PERMISSION:
            return DCGM_INT32_NOT_PERMISSIONED;

        case LWML_ERROR_NOT_FOUND:
            return DCGM_INT32_NOT_FOUND;

        default:
            return DCGM_INT32_BLANK;
    }

    return DCGM_INT32_BLANK;
}

/*****************************************************************************/
double DcgmCacheManager::LwmlErrorToDoubleValue(lwmlReturn_t lwmlReturn)
{
    switch(lwmlReturn)
    {
        case LWML_SUCCESS:
            LW_ASSERT(0); /* Should never get here */
            break;

        case LWML_ERROR_NOT_SUPPORTED:
            return DCGM_FP64_NOT_SUPPORTED;

        case LWML_ERROR_NO_PERMISSION:
            return DCGM_FP64_NOT_PERMISSIONED;

        case LWML_ERROR_NOT_FOUND:
            return DCGM_FP64_NOT_FOUND;

        default:
            return DCGM_FP64_BLANK;
    }

    return DCGM_FP64_BLANK;
}

/*****************************************************************************/
void DcgmCacheManager::EventThreadMain(DcgmCacheManagerEventThread *eventThread)
{
    lwmlReturn_t lwmlReturn;
    lwmlEventData_t eventData = {0};
    unsigned int lwmlGpuIndex;
    dcgm_field_meta_p fieldMeta = 0;
    dcgmcm_watch_info_p watchInfo = 0;
    timelib64_t now, expireTime, threadStartTime;
    int st;
    unsigned int gpuId;
    int numErrors = 0;
    unsigned int timeoutMs = 0; //Do not block in the LWML event call
    dcgmcm_update_thread_t threadCtx;

    InitAndClearThreadCtx(&threadCtx);

    if(!m_lwmlEventSetInitialized)
    {
        PRINT_ERROR("", "event set not initialized");
        Stop(); /* Skip the next loop */
    }

    threadStartTime = timelib_usecSince1970();

    while(!eventThread->ShouldStop())
    {
        /* Clear fvBuffer if it exists */
        ClearThreadCtx(&threadCtx);

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if(!threadCtx.fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx.fvBuffer = new DcgmFvBuffer();
            if(!threadCtx.fvBuffer)
                PRINT_ERROR("", "Got NULL fvBuffer");
        }
            
        MarkEnteredDriver();

        if (!m_lwmlEventSetInitialized)
        {
            MarkReturnedFromDriver();
            sleep(1);
            continue;
        }
        
        /* Try to read an event */
        lwmlReturn = lwmlEventSetWait(m_lwmlEventSet, &eventData, timeoutMs);
        if(lwmlReturn == LWML_ERROR_TIMEOUT)
        {
            PRINT_DEBUG("", "lwmlEventSetWait timeout.");
            MarkReturnedFromDriver();
            sleep(1);
            continue; /* We expect to get this 99.9% of the time. Keep on reading */
        }
        else if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_WARNING("%d", "Got st %d from lwmlEventSetWait", (int)lwmlReturn);
            numErrors++;
            if(numErrors >= 1000)
            {
                /* If we get an excessive number of errors, quit instead of spinning in a hot loop 
                   this will cripple event reading, but it will prevent DCGM from using 100% CPU */
                PRINT_CRITICAL("%d", "Quitting EventThreadMain() after %d errors.", numErrors);
            }
            MarkReturnedFromDriver();
            sleep(1);
            continue;
        }

        now = timelib_usecSince1970();

        lwmlReturn = lwmlDeviceGetIndex(eventData.device, &lwmlGpuIndex);
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_WARNING("", "Unable to colwert device handle to index");
            MarkReturnedFromDriver();
            sleep(1);
            continue;
        }

        gpuId = LwmlIndexToGpuId(lwmlGpuIndex);

        PRINT_DEBUG("%llu %u", "Got lwmlEvent %llu for gpuId %u", eventData.eventType, gpuId);

        switch(eventData.eventType)
        {
            case lwmlEventTypeXidCriticalError:
                fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_XID_ERRORS);
                watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_XID_ERRORS, 1);
                threadCtx.entityKey = watchInfo->watchKey;
                threadCtx.watchInfo = watchInfo;
                expireTime = 0;
                if(watchInfo->maxAgeUsec)
                    expireTime = now - watchInfo->maxAgeUsec;

                /* Only update once we have a valid watchInfo. This is always LWML_SUCCESS
                * because of the for loop condition */
                watchInfo->lastStatus = lwmlReturn;

                AppendEntityInt64(&threadCtx, (long long)eventData.eventData, 0,
                                  now, expireTime);
                break;

            case lwmlEventTypeLWLinkRecoveryError:
            case lwmlEventTypeLWLinkFatalError:                
                fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_LWLINK_ERRORS);
                watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_GPU_LWLINK_ERRORS, 1);
                threadCtx.entityKey = watchInfo->watchKey;
                threadCtx.watchInfo = watchInfo;
                expireTime = 0;
                if(watchInfo->maxAgeUsec)
                    expireTime = now - watchInfo->maxAgeUsec;

                /* Only update once we have a valid watchInfo. This is always LWML_SUCCESS
                * because of the for loop condition */
                watchInfo->lastStatus = lwmlReturn;
                // update eventData with DCGM error type
                eventData.eventData = LwmlGpuLWLinkErrorToDcgmError(eventData.eventType);
                AppendEntityInt64(&threadCtx, (long long)eventData.eventData, 0,
                                  now, expireTime);
                break;

            default:
                PRINT_WARNING("%llX", "Unhandled event type %llX", eventData.eventType);
                break;
        
        }

        if(threadCtx.fvBuffer)
            UpdateFvSubscribers(&threadCtx);
        
        MarkReturnedFromDriver();

        sleep(1);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateTopologyAffinity(dcgmAffinity_t &affinity)
{
    unsigned int elementsFilled = 0;

    for (unsigned int index = 0; index < m_numGpus; index++)
    {
        if (m_gpus[index].status == DcgmcmGpuStatusDetached)
            continue;

        lwmlReturn_t lwmlReturn = lwmlDeviceGetCpuAffinity(m_gpus[index].lwmlDevice, DCGM_AFFINITY_BITMASK_ARRAY_SIZE, affinity.affinityMasks[elementsFilled].bitmask);
        if (LWML_SUCCESS!= lwmlReturn)
            return LwmlReturnToDcgmReturn(lwmlReturn);
        affinity.affinityMasks[elementsFilled].dcgmGpuId = LwmlIndexToGpuId(index);

        elementsFilled++;
    }
    affinity.numGpus = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyAffinity(dcgmcm_update_thread_t *threadCtx, 
                                                     timelib64_t now, 
                                                     timelib64_t expireTime)
{
    dcgmAffinity_t affinity = { 0 };
    PopulateTopologyAffinity(affinity);

    AppendEntityBlob(threadCtx, &affinity, sizeof(dcgmAffinity_t), now, expireTime);
    
    return DCGM_ST_OK;
} 

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetActiveLwSwitchLwLinkCountsForAllGpus(unsigned int *gpuCounts)
{
    lwmlFieldValue_t value = {0};
    lwmlReturn_t lwmlReturn;

    memset(gpuCounts, 0, sizeof(gpuCounts[0])*m_numGpus);

    int gpuCountsIndex = 0;
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmcmGpuStatusDetached)
            continue;

        // Check for LWSwitch connectivity.
        // We assume all-to-all connectivity in presence of LWSwitch.
        // Commenting out the constant name as it's being removed from LWML
        value.fieldId = 1; // LWML_INT_FI_DEV_LWSWITCH_CONNECTED_LINK_COUNT;

        lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal, DeviceGetInternalFieldValues,
                                    (m_gpus[i].lwmlDevice, 1, &value));
        if(lwmlReturn != LWML_SUCCESS)
        {
            PRINT_DEBUG("%d %d", "DeviceGetInternalFieldValues of gpu %d failed with %d. Is the driver >= r400?",
                        m_gpus[i].gpuId, lwmlReturn);
            return DCGM_ST_NOT_SUPPORTED;
        }
        else if (value.lwmlReturn != LWML_SUCCESS)
        {
            PRINT_DEBUG("%d %d", "LwSwitch link count returned lwml status %d for gpu %d",
                        lwmlReturn, m_gpus[i].gpuId);
            gpuCountsIndex++;
            continue;
        }

        gpuCounts[gpuCountsIndex] = value.value.uiVal;
        PRINT_DEBUG("%d %u", "GPU %d has %u active LwSwitch LwLinks.", m_gpus[i].gpuId, value.value.uiVal);
        gpuCountsIndex++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateTopologyLwLink(dcgmTopology_t **topology_pp, unsigned int &topologySize)
{
    dcgmTopology_t *topology_p;
    unsigned int elementArraySize = 0;
    unsigned int elementsFilled = 0;
    unsigned int gpuLwSwitchLinkCounts[DCGM_MAX_NUM_DEVICES];
    dcgmLwLinkStatus_v2 linkStatus;

    if (m_numGpus < 2)
    {
        PRINT_DEBUG("", "Two devices not detected on this system");
        return (DCGM_ST_NOT_SUPPORTED);
    }

    /* Find out how many LwSwitches each GPU is connected to */
    GetActiveLwSwitchLwLinkCountsForAllGpus(gpuLwSwitchLinkCounts);

    // arithmetic series formula to calc number of combinations
    elementArraySize = (unsigned int)((float)(m_numGpus - 1.0) * (1.0 + ((float)m_numGpus - 2.0)/2.0));

    // this is intended to minimize how much we're storing since we rarely will need all 120 entries in the element array
    topologySize = sizeof(dcgmTopology_t) - (sizeof(dcgmTopologyElement_t)*DCGM_TOPOLOGY_MAX_ELEMENTS)
            + elementArraySize * sizeof(dcgmTopologyElement_t);

    topology_p = (dcgmTopology_t *) calloc(1, topologySize);

    *topology_pp = topology_p;

    /* Get the status of each GPU's LwLinks. We will use this to generate the link mask when gpuLwSwitchLinkCounts[x] is nonzero */
    memset(&linkStatus, 0, sizeof(linkStatus));
    PopulateLwLinkLinkStatus(linkStatus);

    topology_p->version = dcgmTopology_version1;
    for (unsigned int index1 = 0; index1 < m_numGpus; index1++)
    {
        if (m_gpus[index1].status == DcgmcmGpuStatusDetached)
            continue;

        int gpuId1 = m_gpus[index1].gpuId;

        if (m_gpus[index1].arch < LWML_CHIP_ARCH_PASCAL) // bracket this when LWLINK becomes not available on an arch
        {
            PRINT_DEBUG("%d", "GPU %d is older than Pascal", gpuId1);
            continue;
        }

        for (unsigned int index2 = index1+1; index2 < m_numGpus; index2++)
        {
            if (m_gpus[index2].status == DcgmcmGpuStatusDetached)
                continue;

            int gpuId2 = m_gpus[index2].gpuId;

            if (m_gpus[index2].arch < LWML_CHIP_ARCH_PASCAL) // bracket this when LWLINK becomes not available on an arch
            {
                PRINT_DEBUG("", "GPU is older than Pascal");
                continue;
            }

            // all of the paths are stored low GPU to higher GPU (i.e. 0 -> 1, 0 -> 2, 1 -> 2, etc.)
            // so for LWLINK though the quantity of links will be the same as determined by querying
            // node 0 or node 1, the link numbers themselves will be different.  Need to store both values.
            unsigned int localLwLinkQuantity = 0, localLwLinkMask = 0;
            unsigned int remoteLwLinkMask = 0;

            // Assign here instead of 6x below
            localLwLinkQuantity = gpuLwSwitchLinkCounts[gpuId1];
            
            // fill in localLwLink information
            for (unsigned localLwLink = 0; localLwLink < LWML_LWLINK_MAX_LINKS; localLwLink++)
            {
                /* If we have LwSwitches, those are are only connections */
                if(gpuLwSwitchLinkCounts[gpuId1] > 0)
                {
                    if(linkStatus.gpus[gpuId1].linkState[localLwLink] == DcgmLwLinkLinkStateUp)
                        localLwLinkMask |= 1 << localLwLink;
                }
                else
                {
                    lwmlPciInfo_t tempPciInfo;
                    lwmlReturn_t lwmlReturn = lwmlDeviceGetLwLinkRemotePciInfo(m_gpus[index1].lwmlDevice, localLwLink, &tempPciInfo);

                    /* If the link is not supported, continue with other links */
                    if (LWML_ERROR_NOT_SUPPORTED == lwmlReturn)
                    {
                        PRINT_DEBUG("%d %d", "GPU %d LWLINK %d not supported", gpuId1, localLwLink);
                        continue;
                    }
                    else if (LWML_SUCCESS != lwmlReturn)
                    {
                        PRINT_DEBUG("%d %d %d", "Unable to retrieve remote PCI info for GPU %d on LWLINK %d. Returns %d", 
                                gpuId1, localLwLink, lwmlReturn);
                        return LwmlReturnToDcgmReturn(lwmlReturn);
                    }
                    if (!strcasecmp(tempPciInfo.busId, m_gpus[index2].pciInfo.busId))
                    {
                        localLwLinkQuantity++;
                        localLwLinkMask |= 1 << localLwLink;
                    }
                }
            }
             
            // fill in remoteLwLink information
            for (unsigned remoteLwLink = 0; remoteLwLink < LWML_LWLINK_MAX_LINKS; remoteLwLink++)
            {
                /* If we have LwSwitches, those are are only connections */
                if(gpuLwSwitchLinkCounts[gpuId2] > 0)
                {
                    if(linkStatus.gpus[gpuId2].linkState[remoteLwLink] == DcgmLwLinkLinkStateUp)
                        remoteLwLinkMask |= 1 << remoteLwLink;
                }
                else
                {
                    lwmlPciInfo_t tempPciInfo;
                    lwmlReturn_t lwmlReturn = lwmlDeviceGetLwLinkRemotePciInfo(m_gpus[index2].lwmlDevice, remoteLwLink, &tempPciInfo);
                    
                    /* If the link is not supported, continue with other links */
                    if (LWML_ERROR_NOT_SUPPORTED == lwmlReturn)
                    {
                        PRINT_DEBUG("%d %d", "GPU %d LWLINK %d not supported", gpuId1, remoteLwLink);
                        continue;
                    }                        
                    else if (LWML_SUCCESS != lwmlReturn)
                    {
                        PRINT_DEBUG("%d %d %d", "Unable to retrieve remote PCI info for GPU %d on LWLINK %d. Returns %d", 
                                gpuId2, remoteLwLink, lwmlReturn);
                        return LwmlReturnToDcgmReturn(lwmlReturn);
                    }
                    if (!strcasecmp(tempPciInfo.busId, m_gpus[index1].pciInfo.busId))
                    {
                        remoteLwLinkMask |= 1 << remoteLwLink;
                    }
                }
            }
             
            topology_p->element[elementsFilled].dcgmGpuA = gpuId1;
            topology_p->element[elementsFilled].dcgmGpuB = gpuId2;
            topology_p->element[elementsFilled].AtoBLwLinkIds = localLwLinkMask;
            topology_p->element[elementsFilled].BtoALwLinkIds = remoteLwLinkMask;

            // LWLINK information for path resides in bits 31:8 so it can fold into the PCI path
            // easily
            topology_p->element[elementsFilled].path = (dcgmGpuTopologyLevel_t) ((1 << (localLwLinkQuantity - 1)) << 8);
            elementsFilled++;
        }
    }

    topology_p->numElements = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyLwLink(dcgmcm_update_thread_t *threadCtx, 
                                                   timelib64_t now, timelib64_t expireTime)
{
    dcgmTopology_t *topology_p = NULL;
    unsigned int topologySize = 0;
    dcgmReturn_t ret = PopulateTopologyLwLink(&topology_p, topologySize);

    if (ret == DCGM_ST_NOT_SUPPORTED && threadCtx->watchInfo)
        threadCtx->watchInfo->lastStatus = LWML_ERROR_NOT_SUPPORTED;
    
    AppendEntityBlob(threadCtx, topology_p, topologySize, now, expireTime);

    if (topology_p != NULL)
        free(topology_p);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::BufferOrCacheLatestVgpuValue(dcgmcm_update_thread_t *threadCtx,
                                                            lwmlVgpuInstance_t vgpuId, 
                                                            dcgm_field_meta_p fieldMeta)
{
    timelib64_t now, expireTime;
    lwmlReturn_t lwmlReturn;

    if(!threadCtx || !fieldMeta || fieldMeta->scope != DCGM_FS_DEVICE)
        return DCGM_ST_BADPARAM;

    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    now = timelib_usecSince1970();

    /* Expiration is either measured in absolute time or 0 */
    expireTime = 0;
    if(watchInfo && watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Set without lock before we possibly return on error so we don't get in a hot
     * polling loop on something that is unsupported or errors. Not getting the lock
     * ahead of time because we don't want to hold the lock across a driver call that
     * could be long */
    if(watchInfo)
        watchInfo->lastQueriedUsec = now;

    switch(fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_VGPU_VM_ID:
        case DCGM_FI_DEV_VGPU_VM_NAME:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;
            lwmlVgpuVmIdType_t vmIdType;
            char vmName[DCGM_DEVICE_UUID_BUFFER_SIZE];
            FILE *fp;
            char cmd[128], tmp_name[DCGM_DEVICE_UUID_BUFFER_SIZE];

            lwmlReturn = lwmlVgpuInstanceGetVmID(vgpuId, buffer, bufferSize, &vmIdType);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(fieldMeta->fieldId == DCGM_FI_DEV_VGPU_VM_ID)
            {
                AppendEntityString(threadCtx, buffer, now, expireTime);
            }
            else
            {
#if defined(LW_VMWARE)
                /* Command exelwted is specific to VMware */
                snprintf(cmd, sizeof(cmd), "localcli vm process list | grep \"World ID: %s\" -B 1 | head -1 | cut -f1 -d ':'", buffer);

                if(strlen(cmd)==0)
                    return DCGM_ST_NO_DATA;

                if (NULL == (fp = popen(cmd,"r")))
                {
                    lwmlReturn = LWML_ERROR_NOT_FOUND;
                    AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now,
                                       expireTime);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
                if (fgets(tmp_name, sizeof(tmp_name), fp))
                {
                    char *eol = strchr(tmp_name, '\n');
                    if (eol)
                        *eol = 0;
                    AppendEntityString(threadCtx, tmp_name, now, expireTime);
                }
                else
                {
                    lwmlReturn = LWML_ERROR_NOT_FOUND;
                    AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                }
                pclose(fp);
#else
                /* Soon to be implemented for other elwironments. Appending error string for now. */
                lwmlReturn = LWML_ERROR_NOT_SUPPORTED;
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
#endif
            }
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE:
        {
            unsigned int vgpuTypeId = 0;

            lwmlReturn = lwmlVgpuInstanceGetType(vgpuId, &vgpuTypeId);
            if (lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetType failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, vgpuTypeId, 0,
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_UUID:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            lwmlReturn = lwmlVgpuInstanceGetUUID(vgpuId, buffer, bufferSize);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetUUID failed with status %d for vgpuId %u",
                            (int)lwmlReturn, vgpuId);
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now,
                                   expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityString(threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            lwmlReturn = lwmlVgpuInstanceGetVmDriverVersion(vgpuId, buffer, bufferSize);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetVmDriverVersion failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(strcmp("Not Available", buffer))
            {
                /* Updating the cache frequency to once every 15 minutes after a known driver version is fetched. */
                if(watchInfo && watchInfo->monitorFrequencyUsec != 900000000)
                {
                    dcgmReturn_t status = UpdateFieldWatch(watchInfo, 900000000, 900.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    if (DCGM_ST_OK != status)
                    {
                        PRINT_ERROR("%d %u", "UpdateFieldWatch failed for vgpuId %u and fieldId %d", vgpuId, fieldMeta->fieldId);
                        return status;
                    }
                }
            }

            AppendEntityString(threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
        {
            unsigned long long fbUsage;

            lwmlReturn = lwmlVgpuInstanceGetFbUsage(vgpuId, &fbUsage);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetFbUsage failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            fbUsage = fbUsage / (1024 * 1024);
            AppendEntityInt64(threadCtx, fbUsage, 0,
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_LICENSE_STATUS:
        {
            unsigned int licenseState;

            lwmlReturn = lwmlVgpuInstanceGetLicenseStatus(vgpuId, &licenseState);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetLicenseStatus failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(licenseState == 1)
            {
                /* Updating the cache frequency to once every 15 minutes when VM is licensed. */
                if(watchInfo && watchInfo->monitorFrequencyUsec != 900000000)
                {
                    dcgmReturn_t status = UpdateFieldWatch(watchInfo, 900000000, 900.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    if (DCGM_ST_OK != status)
                    {
                        PRINT_ERROR("%d %u", "UpdateFieldWatch failed for vgpuId %u and fieldId %d", vgpuId, fieldMeta->fieldId);
                        return status;
                    }
                }
            }
            else if (licenseState == 0 && (watchInfo && watchInfo->monitorFrequencyUsec != 1000000))
            {
                /* Updating the cache frequency to once every 1 sec, when VM is unlicensed and current caching frequency is 15 mins. */
                dcgmReturn_t status = UpdateFieldWatch(watchInfo, 1000000, 600.0, 600, DcgmWatcher(DcgmWatcherTypeCacheManager));
                if (DCGM_ST_OK != status)
                {
                    PRINT_ERROR("%d %u", "UpdateFieldWatch failed for vgpuId %u and fieldId %d", vgpuId, fieldMeta->fieldId);
                    return status;
                }
            }

            AppendEntityInt64(threadCtx, licenseState, 0,
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT:
        {
            unsigned int frameRateLimit;

            lwmlReturn = lwmlVgpuInstanceGetFrameRateLimit(vgpuId, &frameRateLimit);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetFrameRateLimit failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, frameRateLimit, 0,
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_PCI_ID:
        {
            char vgpuPciId[LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
            unsigned int length = LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE;

            lwmlReturn = lwmlVgpuInstanceGetGpuPciId(vgpuId, vgpuPciId, &length);
            if (watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn && LWML_ERROR_DRIVER_NOT_LOADED != lwmlReturn)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetGpuPciId failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Updating the cache frequency to once every 60 minutes after a valid vGPU PCI Id is fetched. */
            if ((strcmp("00000000:00:00.0", vgpuPciId) != 0) && (watchInfo && watchInfo->monitorFrequencyUsec != 3600000000))
            {
                UpdateFieldWatch(watchInfo, 3600000000, 3600.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                PRINT_ERROR("%d %u", "UpdateFieldWatch failed for vgpuId %u and fieldId %d", vgpuId, fieldMeta->fieldId);
            }

            AppendEntityString(threadCtx, vgpuPciId, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_STATS:
        {
            dcgmDeviceEncStats_t vgpuEncStats;

            lwmlReturn = lwmlVgpuInstanceGetEncoderStats(vgpuId, &vgpuEncStats.sessionCount, &vgpuEncStats.averageFps,
                            &vgpuEncStats.averageLatency);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                memset(&vgpuEncStats, 0, sizeof(vgpuEncStats));
                AppendEntityBlob(threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)),
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityBlob(threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)),
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO:
        {
            dcgmDeviceVgpuEncSessions_t *vgpuEncSessionsInfo = NULL;
            lwmlEncoderSessionInfo_t *sessionInfo = NULL;
            unsigned int i, sessionCount = 0;

            lwmlReturn = lwmlVgpuInstanceGetEncoderSessions(vgpuId, &sessionCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            sessionInfo = (lwmlEncoderSessionInfo_t *)malloc(sizeof(*sessionInfo)*(sessionCount));
            if(!sessionInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo)*(sessionCount)));
                return DCGM_ST_MEMORY;
            }

            vgpuEncSessionsInfo = (dcgmDeviceVgpuEncSessions_t *)malloc(sizeof(*vgpuEncSessionsInfo)*(sessionCount+1));
            if(!vgpuEncSessionsInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuEncSessionsInfo)*(sessionCount+1)));
                free(sessionInfo);
                return DCGM_ST_MEMORY;
            }

            if (lwmlReturn != LWML_SUCCESS)
            {
                vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = 0;
                AppendEntityBlob(threadCtx, vgpuEncSessionsInfo, (int)(sizeof(*vgpuEncSessionsInfo)*(sessionCount+1)),
                                  now, expireTime);
                free(sessionInfo);
                free(vgpuEncSessionsInfo);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if (sessionCount != 0)
            {
                lwmlReturn = lwmlVgpuInstanceGetEncoderSessions(vgpuId, &sessionCount, sessionInfo);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                if (lwmlReturn != LWML_SUCCESS)
                {
                    PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetEncoderSessions failed with status %d for vgpuId %d",
                                  (int)lwmlReturn, vgpuId);
                    free(sessionInfo);
                    free(vgpuEncSessionsInfo);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }

            /* First element of the array holds the count */
            vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = sessionCount;

            for (i = 0; i < sessionCount; i++)
            {
                vgpuEncSessionsInfo[i+1].encoderSessionInfo.vgpuId = sessionInfo[i].vgpuInstance;
                vgpuEncSessionsInfo[i+1].sessionId      = sessionInfo[i].sessionId;
                vgpuEncSessionsInfo[i+1].pid            = sessionInfo[i].pid;
                vgpuEncSessionsInfo[i+1].codecType      = (dcgmEncoderType_t)sessionInfo[i].codecType;
                vgpuEncSessionsInfo[i+1].hResolution    = sessionInfo[i].hResolution;
                vgpuEncSessionsInfo[i+1].vResolution    = sessionInfo[i].vResolution;
                vgpuEncSessionsInfo[i+1].averageFps     = sessionInfo[i].averageFps;
                vgpuEncSessionsInfo[i+1].averageLatency = sessionInfo[i].averageLatency;
            }
            AppendEntityBlob(threadCtx, vgpuEncSessionsInfo, (int)(sizeof(*vgpuEncSessionsInfo)*(sessionCount+1)),
                              now, expireTime);
            free(sessionInfo);
            free(vgpuEncSessionsInfo);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_STATS:
        {
            dcgmDeviceFbcStats_t vgpuFbcStats;
            lwmlFBCStats_t fbcStats;

            lwmlReturn = lwmlVgpuInstanceGetFBCStats(vgpuId, &fbcStats);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetFBCStats failed with status %d for vgpuId %u",
                              (int)lwmlReturn, vgpuId);
                memset(&vgpuFbcStats, 0, sizeof(vgpuFbcStats));
                AppendEntityBlob(threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)),
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            vgpuFbcStats.version = dcgmDeviceFbcStats_version;
            vgpuFbcStats.sessionCount = fbcStats.sessionsCount;
            vgpuFbcStats.averageFps = fbcStats.averageFPS;
            vgpuFbcStats.averageLatency = fbcStats.averageLatency;

            AppendEntityBlob(threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)),
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO:
        {
            dcgmReturn_t status = GetVgpuInstanceFBCSessionsInfo(vgpuId, threadCtx, watchInfo, now, expireTime);
            if (DCGM_ST_OK != status)
                return status;
            break;
        }

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

        return DCGM_ST_OK;
}

/*****************************************************************************/
/* vGPU Index key space for gpuId */
#define LWCMCM_START_VGPU_IDX_FOR_GPU(gpuId)    (gpuId * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)
#define LWCMCM_END_VGPU_IDX_FOR_GPU(gpuId)      ((gpuId+1) * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::BufferOrCacheLatestGpuValue(dcgmcm_update_thread_t *threadCtx, 
                                                           dcgm_field_meta_p fieldMeta)
{
    int st;
    timelib64_t now, expireTime, lastCollectionTimeStamp = 0;
    lwmlReturn_t lwmlReturn;
    lwmlDevice_t lwmlDevice = 0;

    if(!threadCtx || !fieldMeta)
        return DCGM_ST_BADPARAM;

    unsigned int gpuId = threadCtx->entityKey.entityId; /* Only valid for GPU fields */
    if (gpuId >= m_numGpus)
        return DCGM_ST_GENERIC_ERROR;

    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    /* Do we need a device handle? */
    if(fieldMeta->scope != DCGM_FS_GLOBAL)
        lwmlDevice = m_gpus[gpuId].lwmlDevice;

    now = timelib_usecSince1970();

    /* Expiration is either measured in absolute time or 0 */
    expireTime = 0;
    if(watchInfo && watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Set without lock before we possibly return on error so we don't get in a hot
     * polling loop on something that is unsupported or errors. Not getting the lock
     * ahead of time because we don't want to hold the lock across a driver call that
     * could be long */
    if(watchInfo)
    {
        lastCollectionTimeStamp = watchInfo->lastQueriedUsec;
        watchInfo->lastQueriedUsec = now;
    }

    switch(fieldMeta->fieldId)
    {
        case DCGM_FI_DRIVER_VERSION:
        {
            char buf[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {0};
            lwmlReturn = lwmlSystemGetDriverVersion(buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now,
                                   expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_LWML_VERSION:
        {
            char buf[LWML_SYSTEM_LWML_VERSION_BUFFER_SIZE] = {0};
            lwmlReturn = lwmlSystemGetLWMLVersion(buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now,
                                   expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_PROCESS_NAME:
        {
            char buf[128] = {0};
            lwmlReturn = lwmlSystemGetProcessName((unsigned int)lwosProcessId(), buf, sizeof(buf)-1);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now,
                                   expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_COUNT:
        {
            unsigned int deviceCount = 0;

            lwmlReturn = lwmlDeviceGetCount(&deviceCount);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)deviceCount, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_NAME:
        {
            char buf[LWML_DEVICE_NAME_BUFFER_SIZE] = {0};
            lwmlReturn = lwmlDeviceGetName(lwmlDevice, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_BRAND:
        {
            lwmlBrandType_t brand;
            char *brandString = 0;
            lwmlReturn = lwmlDeviceGetBrand(lwmlDevice, &brand);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* String constants stolen from lwsmi/reporting.c. Using switch instead
             * of table in case more are added */
            switch(brand)
            {
                case LWML_BRAND_QUADRO:
                    brandString = (char *)"Lwdqro";
                    break;
                case LWML_BRAND_TESLA:
                    brandString = (char *)"Tesla";
                    break;
                case LWML_BRAND_LWS:
                    brandString = (char *)"LWS";
                    break;
                case LWML_BRAND_GRID:
                    brandString = (char *)"Grid";
                    break;
                case LWML_BRAND_LWIDIA_VAPPS:
                    brandString = (char *)"LWPU Virtual Applications";
                    break;
                case LWML_BRAND_LWIDIA_VPC:
                    brandString = (char *)"LWPU Virtual PC";
                    break;
                case LWML_BRAND_LWIDIA_VCS:
                    brandString = (char *)"LWPU Virtual Compute Server";
                    break;
                case LWML_BRAND_LWIDIA_VWS:
                    brandString = (char *)"LWPU RTX Virtual Workstation";
                    break;
                case LWML_BRAND_LWIDIA_CLOUD_GAMING:
                    brandString = (char *)"LWPU Cloud Gaming";
                    break;
                case LWML_BRAND_GEFORCE:
                    brandString = (char *)"VdChip";
                    break;
                case LWML_BRAND_TITAN:
                    brandString = (char *)"Titan";
                    break;
                case LWML_BRAND_QUADRO_RTX:
                    brandString = (char *)"Lwdqro RTX";
                    break;
                case LWML_BRAND_LWIDIA_RTX:
                    brandString = (char *)"LWPU RTX";
                    break;
                case LWML_BRAND_LWIDIA:
                    brandString = (char *)"LWPU";
                    break;
                case LWML_BRAND_GEFORCE_RTX:
                    brandString = (char *)"VdChip RTX";
                    break;
                case LWML_BRAND_TITAN_RTX:
                    brandString = (char *)"TITAN RTX";
                    break;
                case LWML_BRAND_UNKNOWN:
                default:
                    brandString = (char *)"Unknown";
                    break;
            }

            AppendEntityString(threadCtx, brandString, now,
                               expireTime);
            break;
        }

        case DCGM_FI_DEV_LWML_INDEX:
        {
            /* There's really no point in making the call since we passed in what we want */
            if(watchInfo)
                watchInfo->lastStatus = LWML_SUCCESS;
            AppendEntityInt64(threadCtx, GpuIdToLwmlIndex(gpuId), 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_SERIAL:
        {
            char buf[LWML_DEVICE_SERIAL_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetSerial(lwmlDevice, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_CPU_AFFINITY_0:
        case DCGM_FI_DEV_CPU_AFFINITY_1:
        case DCGM_FI_DEV_CPU_AFFINITY_2:
        case DCGM_FI_DEV_CPU_AFFINITY_3:
        {

            long long saveValue = 0;
            long long values[4] = {0};
            unsigned int Nlongs = (sizeof(long) == 8) ? 4 : 8;
            int affinityIndex = fieldMeta->fieldId - DCGM_FI_DEV_CPU_AFFINITY_0;

            lwmlReturn = lwmlDeviceGetCpuAffinity(lwmlDevice, Nlongs, (unsigned long *)&values[0]);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Save the value that corresponds with the requested field */
            saveValue = values[affinityIndex];

            AppendEntityInt64(threadCtx, saveValue, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_UUID:
        {
            char buf[DCGM_DEVICE_UUID_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetUUID(lwmlDevice, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MINOR_NUMBER:
        {
            unsigned int minorNumber = 0;
            lwmlReturn = lwmlDeviceGetMinorNumber(lwmlDevice, &minorNumber);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)minorNumber, 1, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_LWDA_COMPUTE_CAPABILITY:
        {
            int major = 0;
            int minor = 0;
            long long ccc = 0;
            lwmlReturn = lwmlDeviceGetLwdaComputeCapability(lwmlDevice, &major, &minor);
            if (watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return (LwmlReturnToDcgmReturn(lwmlReturn));
            }

            // Store the major version in the upper 16 bits, and the minor version in the lower 16 bits
            ccc = (major << 16) | minor;
            AppendEntityInt64(threadCtx, ccc, 1, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_OEM_INFOROM_VER:
        {
            char buf[LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetInforomVersion(lwmlDevice, LWML_INFOROM_OEM, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_ECC_INFOROM_VER:
        {
            char buf[LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetInforomVersion(lwmlDevice, LWML_INFOROM_ECC, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_POWER_INFOROM_VER:
        {
            char buf[LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetInforomVersion(lwmlDevice, LWML_INFOROM_POWER, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_IMAGE_VER:
        {
            char buf[LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {0};

            lwmlReturn = lwmlDeviceGetInforomImageVersion(lwmlDevice, buf, sizeof(buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn), now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_CONFIG_CHECK:
        {
            unsigned int checksum = 0;

            lwmlReturn = lwmlDeviceGetInforomConfigurationChecksum(lwmlDevice, &checksum);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)checksum, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_CONFIG_VALID:
        {
            lwmlReturn = lwmlDeviceValidateInforom(lwmlDevice);

            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS && lwmlReturn != LWML_ERROR_CORRUPTED_INFOROM)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            unsigned long long valid = ((lwmlReturn == LWML_SUCCESS) ? 1 : 0);

            AppendEntityInt64(threadCtx, valid, 0, now,
                              expireTime);
            break;
        }
        
        case DCGM_FI_DEV_VBIOS_VERSION:
        {
            char buf[LWML_DEVICE_VBIOS_VERSION_BUFFER_SIZE] = {0};


            lwmlReturn = lwmlDeviceGetVbiosVersion(lwmlDevice, buf, sizeof (buf));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS) {
                AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                        now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;           
        }
        
        case DCGM_FI_DEV_PCI_BUSID:
        case DCGM_FI_DEV_PCI_COMBINED_ID:
        case DCGM_FI_DEV_PCI_SUBSYS_ID:
        {
            lwmlPciInfo_t pciInfo;

            lwmlReturn = lwmlDeviceGetPciInfo(lwmlDevice, &pciInfo);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                /* Append a blank value of the correct type for our fieldId */
                switch(fieldMeta->fieldType)
                {
                    case DCGM_FT_STRING:
                        AppendEntityString(threadCtx, LwmlErrorToStringValue(lwmlReturn),
                                           now, expireTime);
                        break;
                    case DCGM_FT_INT64:
                        AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn),
                                          0, now, expireTime);
                        break;
                    default:
                        PRINT_ERROR("%c", "Unhandled field type: %c", fieldMeta->fieldType);
                        break;
                }

                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Success. Append the correct value */
            switch(fieldMeta->fieldId)
            {
                case DCGM_FI_DEV_PCI_BUSID:
                    AppendEntityString(threadCtx, pciInfo.busId, now, expireTime);
                    break;
                case DCGM_FI_DEV_PCI_COMBINED_ID:
                    AppendEntityInt64(threadCtx, (long long)pciInfo.pciDeviceId,
                                      0, now, expireTime);
                    break;
                case DCGM_FI_DEV_PCI_SUBSYS_ID:
                    AppendEntityInt64(threadCtx, (long long)pciInfo.pciSubSystemId,
                                      0, now, expireTime);
                    break;
                default:
                    PRINT_ERROR("%d", "Unhandled fieldId %d", (int)fieldMeta->fieldId);
                    break;
            }
            

            break;
        }

        case DCGM_FI_DEV_GPU_TEMP:
        {
            unsigned int tempUint;

            lwmlReturn = lwmlDeviceGetTemperature(lwmlDevice, LWML_TEMPERATURE_GPU, &tempUint);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                                  expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)tempUint, 0, now,
                              expireTime);
            break;
        }
        
        case DCGM_FI_DEV_SLOWDOWN_TEMP: /* Fall through is intentional */
        case DCGM_FI_DEV_SHUTDOWN_TEMP:
        {
            lwmlTemperatureThresholds_t thresholdType;
            unsigned int temp;

            if (fieldMeta->fieldId == DCGM_FI_DEV_SLOWDOWN_TEMP)
                thresholdType = LWML_TEMPERATURE_THRESHOLD_SLOWDOWN;
            else
                thresholdType = LWML_TEMPERATURE_THRESHOLD_SHUTDOWN;

            lwmlReturn = lwmlDeviceGetTemperatureThreshold(lwmlDevice, thresholdType, &temp);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now,
                        expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, temp, 0, now,
                    expireTime);
            
            break;
        }

        case DCGM_FI_DEV_POWER_USAGE:
        {
            unsigned int powerUint;
            double powerDbl;

            lwmlReturn = lwmlDeviceGetPowerUsage(lwmlDevice, &powerUint);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(lwmlReturn), 0,
                                   now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            powerDbl = ((double)powerUint) / 1000.0; /* Colwert to watts */
            AppendEntityDouble(threadCtx, powerDbl, 0,
                               now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SM_CLOCK:
        {
            unsigned int valueI32 = 0;

            lwmlReturn = lwmlDeviceGetClockInfo(lwmlDevice, LWML_CLOCK_SM, &valueI32);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_MEM_CLOCK:
        {
            unsigned int valueI32 = 0;

            lwmlReturn = lwmlDeviceGetClockInfo(lwmlDevice, LWML_CLOCK_MEM, &valueI32);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_VIDEO_CLOCK:
        {
            unsigned int valueI32 = 0;

            lwmlReturn = lwmlDeviceGetClockInfo(lwmlDevice, LWML_CLOCK_VIDEO, &valueI32);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_MAX_SM_CLOCK: /* Intentional fall-through */
        case DCGM_FI_DEV_MAX_MEM_CLOCK: 
        case DCGM_FI_DEV_MAX_VIDEO_CLOCK:
        {
            unsigned int valueI32 = 0;
            lwmlClockType_t clockType;

            if(fieldMeta->fieldId == DCGM_FI_DEV_MAX_SM_CLOCK)
                clockType = LWML_CLOCK_SM;
            else if(fieldMeta->fieldId == DCGM_FI_DEV_MAX_MEM_CLOCK)
                clockType = LWML_CLOCK_MEM;
            else //Assume video clock
                clockType = LWML_CLOCK_VIDEO;

            lwmlReturn = lwmlDeviceGetMaxClockInfo(lwmlDevice, clockType, &valueI32);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_FAN_SPEED:
        {
            unsigned int valueI32 = 0;

            lwmlReturn = lwmlDeviceGetFanSpeed(lwmlDevice, &valueI32);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_TX_THROUGHPUT:
        {
            unsigned int valueI32 = 0;

            /* counters are in 1KB chunks */
            lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                        DeviceGetPcieUtilCounterInternal,
                                        (lwmlDevice, LWML_INT_PCIE_UTIL_TX_BYTES, &valueI32));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_RX_THROUGHPUT:
        {
            unsigned int valueI32 = 0;

            /* counters are in 1KB chunks */
            lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                        DeviceGetPcieUtilCounterInternal,
                                        (lwmlDevice, LWML_INT_PCIE_UTIL_RX_BYTES, &valueI32));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:
        {
            unsigned int counter = 0;

            lwmlReturn = lwmlDeviceGetPcieReplayCounter(lwmlDevice, &counter);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)counter, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_GPU_UTIL:
        case DCGM_FI_DEV_MEM_COPY_UTIL:
        {
            lwmlUtilization_t utilization;
            unsigned int valueI32;

            lwmlReturn = lwmlDeviceGetUtilizationRates(lwmlDevice, &utilization);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(fieldMeta->fieldId == DCGM_FI_DEV_GPU_UTIL)
                valueI32 = utilization.gpu;
            else
                valueI32 = utilization.memory;
            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_ENC_UTIL:
        {
            unsigned int enlwtil;
            unsigned int samplingPeriodUs;

            lwmlReturn = lwmlDeviceGetEncoderUtilization(lwmlDevice, &enlwtil, &samplingPeriodUs);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, enlwtil, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_DEC_UTIL:
        {
            unsigned int delwtil;
            unsigned int samplingPeriodUs;

            lwmlReturn = lwmlDeviceGetDecoderUtilization(lwmlDevice, &delwtil, &samplingPeriodUs);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, delwtil, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_AUTOBOOST:
        {
            lwmlEnableState_t isEnabled, defaultIsEnabled;
            lwmlReturn = lwmlDeviceGetAutoBoostedClocksEnabled(lwmlDevice, &isEnabled, &defaultIsEnabled);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, isEnabled, 0,
                              now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            unsigned int powerLimitInt;
            double powerLimitDbl;

            lwmlReturn = lwmlDeviceGetPowerManagementLimit(lwmlDevice, &powerLimitInt);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }                
            
            powerLimitDbl = powerLimitInt/1000;
            AppendEntityDouble(threadCtx, powerLimitDbl, 0,
                              now, expireTime);            

            break;
        }


        case DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF:
        {
            unsigned int defaultPowerLimitInt;
            double defaultPowerLimitDbl;
            
            lwmlReturn = lwmlDeviceGetPowerManagementDefaultLimit(lwmlDevice, &defaultPowerLimitInt);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            
            defaultPowerLimitDbl = defaultPowerLimitInt/1000;
            AppendEntityDouble(threadCtx, defaultPowerLimitDbl, 0,
                              now, expireTime);             
            
            break;
        }
        
        
        case DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX:  /* fall-through is intentional */
        case DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN:
        {
            unsigned int maxLimitInt, minLimitInt;
            
            lwmlReturn = lwmlDeviceGetPowerManagementLimitConstraints(lwmlDevice, &minLimitInt, &maxLimitInt);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            
            if (fieldMeta->fieldId == DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX) {
                AppendEntityDouble(threadCtx, maxLimitInt/1000, 0,
                        now, expireTime);                
            } else {
                AppendEntityDouble(threadCtx, minLimitInt/1000, 0,
                        now, expireTime);                                
            }
            
            
            break;
        }
        
        case DCGM_FI_DEV_APP_SM_CLOCK:
        {
            unsigned int procClk;

            lwmlReturn = lwmlDeviceGetApplicationsClock(lwmlDevice, LWML_CLOCK_GRAPHICS, &procClk);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }     
            
            AppendEntityInt64(threadCtx, procClk, 0,
                              now, expireTime);             
            
            break;
        }
        
        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            unsigned int memClk;

            lwmlReturn = lwmlDeviceGetApplicationsClock(lwmlDevice, LWML_CLOCK_MEM, &memClk);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }     
            
            AppendEntityInt64(threadCtx, memClk, 0,
                              now, expireTime);
            
            break;
        }

        case DCGM_FI_DEV_CLOCK_THROTTLE_REASONS:
        {
            unsigned long long clockThrottleReasons = 0;
            lwmlReturn = lwmlDeviceGetLwrrentClocksThrottleReasons(lwmlDevice, &clockThrottleReasons);
            if (watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            AppendEntityInt64(threadCtx, clockThrottleReasons, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SUPPORTED_CLOCKS:
        {
            lwmlSupportedClocks_t supportedClocks;

            lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                        DeviceGetSupportedClocks,
                                        (lwmlDevice, &supportedClocks));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            if(lwmlReturn != LWML_SUCCESS)
            {
                /* Zero out the structure. We're still going to insert it */
                memset(&supportedClocks, 0, sizeof(supportedClocks));
            }

            AppendDeviceSupportedClocks(threadCtx, &supportedClocks,
                                        now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_BAR1_TOTAL:
        case DCGM_FI_DEV_BAR1_USED:
        case DCGM_FI_DEV_BAR1_FREE:
        {
            lwmlBAR1Memory_t bar1Memory;
            long long value;

            lwmlReturn = lwmlDeviceGetBAR1MemoryInfo(lwmlDevice, &bar1Memory);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(fieldMeta->fieldId == DCGM_FI_DEV_BAR1_TOTAL)
                value = (long long)bar1Memory.bar1Total;
            else if(fieldMeta->fieldId == DCGM_FI_DEV_BAR1_USED)
                value = (long long)bar1Memory.bar1Used;
            else //DCGM_FI_DEV_BAR1_FREE
                value = (long long)bar1Memory.bar1Free;

            value = value / 1024 / 1024;
            AppendEntityInt64(threadCtx, value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FB_TOTAL:
        case DCGM_FI_DEV_FB_USED:
        case DCGM_FI_DEV_FB_FREE:
        {
            lwmlMemory_t fbMemory;
            unsigned int total, used, free;

            lwmlReturn = lwmlDeviceGetMemoryInfo(lwmlDevice, &fbMemory);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if(fieldMeta->fieldId == DCGM_FI_DEV_FB_TOTAL) {
                total = fbMemory.total / (1024 * 1024);
                AppendEntityInt64(threadCtx, total, 0,
                              now, expireTime);
            }
            else if(fieldMeta->fieldId == DCGM_FI_DEV_FB_USED) {
                used = fbMemory.used / (1024 * 1024);
                AppendEntityInt64(threadCtx, used, 0,
                              now, expireTime);
            } else {
                free = fbMemory.free / (1024 * 1024);
                AppendEntityInt64(threadCtx, free, 0,
                              now, expireTime);
            }
            break;
        }

        case DCGM_FI_DEV_VIRTUAL_MODE:
        {
            lwmlGpuVirtualizationMode_t virtualMode;

            lwmlReturn = lwmlDeviceGetVirtualizationMode(lwmlDevice, &virtualMode);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)virtualMode, 0, now,
                              expireTime);
            break;
        }

        case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
        {
            unsigned int vgpuCount = 0, i;
            lwmlVgpuTypeId_t *supportedVgpuTypeIds = NULL;
            dcgmDeviceVgpuTypeInfo_t *vgpuTypeInfo = NULL;
            lwmlReturn_t errorCode = LWML_SUCCESS;

            lwmlReturn = lwmlDeviceGetSupportedVgpus(lwmlDevice, &vgpuCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            supportedVgpuTypeIds = (lwmlVgpuTypeId_t *)malloc(sizeof(*supportedVgpuTypeIds)*vgpuCount);
            if(!supportedVgpuTypeIds)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*supportedVgpuTypeIds)*vgpuCount));
                return DCGM_ST_MEMORY;
            }

            vgpuTypeInfo = (dcgmDeviceVgpuTypeInfo_t *)malloc(sizeof(*vgpuTypeInfo)*(vgpuCount+1));
            if(!vgpuTypeInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuTypeInfo)*(vgpuCount+1)));
                free(supportedVgpuTypeIds);
                return DCGM_ST_MEMORY;
            }

            if (lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE) {
                vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount = 0;
                AppendEntityBlob(threadCtx, vgpuTypeInfo, (int)(sizeof(*vgpuTypeInfo)*(vgpuCount+1)),
                                  now, expireTime);
                free(supportedVgpuTypeIds);
                free(vgpuTypeInfo);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* First element of the array holds the count */
            vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount = vgpuCount;

            if (vgpuCount != 0) {
                lwmlReturn = lwmlDeviceGetSupportedVgpus(lwmlDevice, &vgpuCount, supportedVgpuTypeIds);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                if(lwmlReturn != LWML_SUCCESS)
                {
                    PRINT_ERROR("%d %d", "lwmlDeviceGetSupportedVgpus failed with status %d for gpuId %u",
                                  (int)lwmlReturn, gpuId);
                    free(supportedVgpuTypeIds);
                    free(vgpuTypeInfo);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }

            for (i = 0; i < vgpuCount; i++) {
                unsigned int nameBufferSize = LWML_VGPU_NAME_BUFFER_SIZE;
                unsigned long long fbTotal;

                vgpuTypeInfo[i+1].vgpuTypeInfo.vgpuTypeId = supportedVgpuTypeIds[i];

                lwmlReturn = lwmlVgpuTypeGetName(supportedVgpuTypeIds[i], vgpuTypeInfo[i+1].vgpuTypeName, &nameBufferSize);
                if(lwmlReturn != LWML_SUCCESS)
                {
                    errorCode = lwmlReturn;
                    strcpy(vgpuTypeInfo[i+1].vgpuTypeName, "Unknown");
                }

                lwmlReturn = lwmlVgpuTypeGetClass(supportedVgpuTypeIds[i], vgpuTypeInfo[i+1].vgpuTypeClass, &nameBufferSize);
                if(lwmlReturn != LWML_SUCCESS)
                {
                    errorCode = lwmlReturn;
                    strcpy(vgpuTypeInfo[i+1].vgpuTypeClass, "Unknown");
                }

                lwmlReturn = lwmlVgpuTypeGetLicense(supportedVgpuTypeIds[i], vgpuTypeInfo[i+1].vgpuTypeLicense, LWML_GRID_LICENSE_BUFFER_SIZE);
                if(lwmlReturn != LWML_SUCCESS)
                {
                    errorCode = lwmlReturn;
                    strcpy(vgpuTypeInfo[i+1].vgpuTypeLicense, "Unknown");
                }

                lwmlReturn = lwmlVgpuTypeGetDeviceID(supportedVgpuTypeIds[i], (unsigned long long *)&vgpuTypeInfo[i+1].deviceId,
                                 (unsigned long long *)&vgpuTypeInfo[i+1].subsystemId);
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].deviceId = -1;
                    vgpuTypeInfo[i+1].subsystemId = -1;
                }

                lwmlReturn = lwmlVgpuTypeGetNumDisplayHeads(supportedVgpuTypeIds[i], (unsigned int *)&vgpuTypeInfo[i+1].numDisplayHeads);
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].numDisplayHeads = -1;
                }

                lwmlReturn = lwmlVgpuTypeGetMaxInstances(lwmlDevice, supportedVgpuTypeIds[i], (unsigned int *)&vgpuTypeInfo[i+1].maxInstances);
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].maxInstances = -1;
                }

                lwmlReturn = lwmlVgpuTypeGetFrameRateLimit(supportedVgpuTypeIds[i], (unsigned int *)&vgpuTypeInfo[i+1].frameRateLimit);
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].frameRateLimit = -1;
                }

                lwmlReturn = lwmlVgpuTypeGetResolution(supportedVgpuTypeIds[i], 0, (unsigned int *)&vgpuTypeInfo[i+1].maxResolutionX,
                                 (unsigned int *)&vgpuTypeInfo[i+1].maxResolutionY);
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].maxResolutionX = -1;
                    vgpuTypeInfo[i+1].maxResolutionY = -1;
                }

                lwmlReturn = lwmlVgpuTypeGetFramebufferSize(supportedVgpuTypeIds[i], &fbTotal);
                fbTotal = fbTotal / (1024 * 1024);
                vgpuTypeInfo[i+1].fbTotal = fbTotal;
                if ((LWML_SUCCESS != lwmlReturn))
                {
                    errorCode = lwmlReturn;
                    vgpuTypeInfo[i+1].fbTotal = -1;
                }
            }

            if(watchInfo)
                watchInfo->lastStatus = errorCode;
            AppendEntityBlob(threadCtx, vgpuTypeInfo, (int)(sizeof(*vgpuTypeInfo)*(vgpuCount+1)),
                          now, expireTime);
            free(supportedVgpuTypeIds);
            free(vgpuTypeInfo);
            break;
        }

        case DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS:
        {
            unsigned int vgpuCount = 0;
            lwmlVgpuTypeId_t *creatableVgpuTypeIds = NULL;

            lwmlReturn = lwmlDeviceGetCreatableVgpus(lwmlDevice, &vgpuCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            // Add 1 to the count because the first spot is used to hold the count
            creatableVgpuTypeIds = (lwmlVgpuTypeId_t *)malloc(sizeof(*creatableVgpuTypeIds)*(vgpuCount + 1));
            if(!creatableVgpuTypeIds)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*creatableVgpuTypeIds)*(vgpuCount+1)));
                return DCGM_ST_MEMORY;
            }

            if (LWML_ERROR_INSUFFICIENT_SIZE != lwmlReturn) {
                creatableVgpuTypeIds[0] = 0;
                AppendEntityBlob(threadCtx, creatableVgpuTypeIds, (int)(sizeof(creatableVgpuTypeIds[0])),
                                 now, expireTime);
                free(creatableVgpuTypeIds);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* First element of the array holds the count */
            creatableVgpuTypeIds[0] = vgpuCount;

            if (vgpuCount != 0) {
                lwmlReturn = lwmlDeviceGetCreatableVgpus(lwmlDevice, &vgpuCount, creatableVgpuTypeIds + 1);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                if(lwmlReturn != LWML_SUCCESS)
                {
                    PRINT_ERROR("%d %u", "lwmlDeviceGetCreatableVgpus failed with status %d for gpuId %u",
                                  (int)lwmlReturn, gpuId);
                    free(creatableVgpuTypeIds);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }

            AppendEntityBlob(threadCtx, creatableVgpuTypeIds, (int)(sizeof(creatableVgpuTypeIds)*(vgpuCount+1)),
                              now, expireTime);

            free(creatableVgpuTypeIds);
            break;
        }

        case DCGM_FI_DEV_VGPU_INSTANCE_IDS:
        {
            unsigned int vgpuCount = 0;
            lwmlVgpuInstance_t *vgpuInstanceIds = NULL;

            lwmlReturn = lwmlDeviceGetActiveVgpus(lwmlDevice, &vgpuCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            vgpuInstanceIds = (lwmlVgpuInstance_t *)malloc(sizeof(vgpuInstanceIds)*(vgpuCount+1));
            if(!vgpuInstanceIds)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuInstanceIds)*vgpuCount));
                return DCGM_ST_MEMORY;
            }

            if ((vgpuCount > 0 && lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE) ||
                (vgpuCount == 0 && lwmlReturn != LWML_SUCCESS))
            {
                vgpuInstanceIds[0] = 0;
                AppendEntityBlob(threadCtx, vgpuInstanceIds, (int)(sizeof(vgpuInstanceIds)),
                                    now, expireTime);
                free(vgpuInstanceIds);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* First element of the array holds the count */
            vgpuInstanceIds[0] = vgpuCount;

            if (vgpuCount != 0) {
                lwmlReturn = lwmlDeviceGetActiveVgpus(lwmlDevice, &vgpuCount, (vgpuInstanceIds+1));
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                if(lwmlReturn != LWML_SUCCESS)
                {
                    PRINT_ERROR("%d %u", "lwmlDeviceGetActiveVgpus failed with status %d for gpuId %u",
                                  (int)lwmlReturn, gpuId);
                    free(vgpuInstanceIds);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }

            AppendEntityBlob(threadCtx, vgpuInstanceIds, (int)(sizeof(vgpuInstanceIds)*(vgpuCount+1)),
                          now, expireTime);

            /* Dynamic handling of add/remove vGPUs */
            ManageVgpuList(gpuId, vgpuInstanceIds);
            free(vgpuInstanceIds);
            break;
        }

        case DCGM_FI_DEV_VGPU_UTILIZATIONS:
        {
            unsigned int vgpuSamplesCount = 0;
            unsigned long long lastSeenTimeStamp = 0;
            lwmlValueType_t sampleValType;
            lwmlVgpuInstanceUtilizationSample_t *vgpuUtilization = NULL;
            dcgmDeviceVgpuUtilInfo_t *vgpuUtilInfo = NULL;

            lwmlReturn = lwmlDeviceGetVgpuUtilization(lwmlDevice, lastSeenTimeStamp, &sampleValType, &vgpuSamplesCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            if ((lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE) &&
                !(lwmlReturn == LWML_SUCCESS && vgpuSamplesCount == 0))
            {
                vgpuUtilInfo = NULL;
                AppendEntityBlob(threadCtx, vgpuUtilInfo, (int)(sizeof(*vgpuUtilInfo)*(vgpuSamplesCount)),
                                 now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            vgpuUtilization = (lwmlVgpuInstanceUtilizationSample_t *)malloc(sizeof(*vgpuUtilization)*vgpuSamplesCount);
            if(!vgpuUtilization)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuUtilization)*vgpuSamplesCount));
                return DCGM_ST_MEMORY;
            }

            lwmlReturn = lwmlDeviceGetVgpuUtilization(lwmlDevice, lastSeenTimeStamp, &sampleValType, &vgpuSamplesCount, vgpuUtilization);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            if((lwmlReturn != LWML_SUCCESS) &&
               !(lwmlReturn == LWML_ERROR_ILWALID_ARGUMENT && vgpuSamplesCount == 0))
            {
                PRINT_WARNING("%d %u", "Unexpected return %d from lwmlDeviceGetVgpuUtilization on gpuId %u",
                              (int)lwmlReturn, gpuId);
                free(vgpuUtilization);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            vgpuUtilInfo = (dcgmDeviceVgpuUtilInfo_t *)malloc(sizeof(*vgpuUtilInfo)*vgpuSamplesCount);
            if(!vgpuUtilInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuUtilization)*vgpuSamplesCount));
                return DCGM_ST_MEMORY;
            }

            for (unsigned int i = 0; i < vgpuSamplesCount; i++)
            {
                vgpuUtilInfo[i].vgpuId          = vgpuUtilization[i].vgpuInstance;
                vgpuUtilInfo[i].smUtil          = vgpuUtilization[i].smUtil.uiVal;
                vgpuUtilInfo[i].memUtil         = vgpuUtilization[i].memUtil.uiVal;
                vgpuUtilInfo[i].enlwtil         = vgpuUtilization[i].enlwtil.uiVal;
                vgpuUtilInfo[i].delwtil         = vgpuUtilization[i].delwtil.uiVal;
            }

            AppendEntityBlob(threadCtx, vgpuUtilInfo, (int)(sizeof(*vgpuUtilInfo)*vgpuSamplesCount),
                             now, expireTime);
            free(vgpuUtilization);
            free(vgpuUtilInfo);
            break;
        }

        case DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION:
        {
            unsigned int vgpuProcessSamplesCount = 0;
            unsigned long long lastSeenTimeStamp = 0;
            lwmlVgpuProcessUtilizationSample_t *vgpuProcessUtilization = NULL;
            dcgmDeviceVgpuProcessUtilInfo_t *vgpuProcessUtilInfo = NULL;

            lwmlReturn = lwmlDeviceGetVgpuProcessUtilization(lwmlDevice, lastSeenTimeStamp, &vgpuProcessSamplesCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            vgpuProcessUtilization = (lwmlVgpuProcessUtilizationSample_t *)malloc(sizeof(*vgpuProcessUtilization)*vgpuProcessSamplesCount);
            if (!vgpuProcessUtilization)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuProcessUtilization)*vgpuProcessSamplesCount));
                return DCGM_ST_MEMORY;
            }

            /* First element of the array holds the vgpuProcessSamplesCount, so allocating memory for (vgpuProcessSamplesCount + 1) elements. */
            vgpuProcessUtilInfo = (dcgmDeviceVgpuProcessUtilInfo_t *)malloc(sizeof(*vgpuProcessUtilInfo)*(vgpuProcessSamplesCount+1));
            if (!vgpuProcessUtilInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuProcessUtilization)*(vgpuProcessSamplesCount+1)));
                free(vgpuProcessUtilization);
                return DCGM_ST_MEMORY;
            }

            if ((lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE) &&
                !(lwmlReturn == LWML_SUCCESS && vgpuProcessSamplesCount == 0))
            {
                vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount  = 0;
                AppendEntityBlob(threadCtx, vgpuProcessUtilInfo, (int)(sizeof(*vgpuProcessUtilInfo)*(vgpuProcessSamplesCount+1)),
                                 now, expireTime);
                free(vgpuProcessUtilization);
                free(vgpuProcessUtilInfo);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if (vgpuProcessSamplesCount != 0)
            {
                lwmlReturn = lwmlDeviceGetVgpuProcessUtilization(lwmlDevice, lastSeenTimeStamp, &vgpuProcessSamplesCount, vgpuProcessUtilization);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;

                if ((lwmlReturn != LWML_SUCCESS) &&
                   !(lwmlReturn == LWML_ERROR_ILWALID_ARGUMENT && vgpuProcessSamplesCount == 0))
                {
                    vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount  = 0;
                    PRINT_WARNING("%d %d", "Unexpected return %d from lwmlDeviceGetVgpuProcessUtilization on gpuId %d",
                                  (int)lwmlReturn, gpuId);
                    free(vgpuProcessUtilization);
                    free(vgpuProcessUtilInfo);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }
            }

            /* First element of the array holds the vgpuProcessSamplesCount */
            vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount  = vgpuProcessSamplesCount;

            for (unsigned int i = 0; i < vgpuProcessSamplesCount; i++)
            {
                vgpuProcessUtilInfo[i+1].vgpuProcessUtilInfo.vgpuId                 = vgpuProcessUtilization[i].vgpuInstance;
                vgpuProcessUtilInfo[i+1].pid                                        = vgpuProcessUtilization[i].pid;
                strncpy(vgpuProcessUtilInfo[i+1].processName, vgpuProcessUtilization[i].processName, DCGM_VGPU_NAME_BUFFER_SIZE-1);
                vgpuProcessUtilInfo[i+1].processName[DCGM_VGPU_NAME_BUFFER_SIZE-1]  = '\0';
                vgpuProcessUtilInfo[i+1].smUtil                                     = vgpuProcessUtilization[i].smUtil;
                vgpuProcessUtilInfo[i+1].memUtil                                    = vgpuProcessUtilization[i].memUtil;
                vgpuProcessUtilInfo[i+1].enlwtil                                    = vgpuProcessUtilization[i].enlwtil;
                vgpuProcessUtilInfo[i+1].delwtil                                    = vgpuProcessUtilization[i].delwtil;
            }

            AppendEntityBlob(threadCtx, vgpuProcessUtilInfo, (int)(sizeof(*vgpuProcessUtilInfo)*(vgpuProcessSamplesCount+1)),
                             now, expireTime);
            free(vgpuProcessUtilization);
            free(vgpuProcessUtilInfo);
            break;
        }

        case DCGM_FI_DEV_ENC_STATS:
        {
            dcgmDeviceEncStats_t devEncStats;

            lwmlReturn = lwmlDeviceGetEncoderStats(lwmlDevice, &devEncStats.sessionCount, &devEncStats.averageFps,
                            &devEncStats.averageLatency);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                memset(&devEncStats, 0, sizeof(devEncStats));
                AppendEntityBlob(threadCtx, &devEncStats, (int)(sizeof(devEncStats)),
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityBlob(threadCtx, &devEncStats, (int)(sizeof(devEncStats)),
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FBC_STATS:
        {
            dcgmDeviceFbcStats_t devFbcStats;
            lwmlFBCStats_t fbcStats;

            lwmlReturn = lwmlDeviceGetFBCStats(lwmlDevice, &fbcStats);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn)
            {
                memset(&devFbcStats, 0, sizeof(devFbcStats));
                AppendEntityBlob(threadCtx, &devFbcStats, (int)(sizeof(devFbcStats)),
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            devFbcStats.version = dcgmDeviceFbcStats_version;
            devFbcStats.sessionCount = fbcStats.sessionsCount;
            devFbcStats.averageFps = fbcStats.averageFPS;
            devFbcStats.averageLatency = fbcStats.averageLatency;
            AppendEntityBlob(threadCtx, &devFbcStats, (int)(sizeof(devFbcStats)),
                              now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FBC_SESSIONS_INFO:
        {
            dcgmReturn_t status = GetDeviceFBCSessionsInfo(lwmlDevice, threadCtx, watchInfo, now, expireTime);
            if (DCGM_ST_OK != status)
                return status;
 
            break;
        }

        case DCGM_FI_DEV_GRAPHICS_PIDS:
        {
            int i;
            unsigned int infoCount = 0;
            lwmlProcessInfo_t *infos = 0;

            /* First, get the capacity we need */
            lwmlReturn = lwmlDeviceGetGraphicsRunningProcesses(lwmlDevice, &infoCount, 0);
            if(lwmlReturn == LWML_SUCCESS)
            {
                PRINT_DEBUG("%u", "No graphics PIDs running on gpuId %u", gpuId);
                break;
            }
            else if(lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE)
            {
                PRINT_WARNING("%d %u", "Unexpected st %d from lwmlDeviceGetGraphicsRunningProcesses on gpuId %u",
                              (int)lwmlReturn, gpuId);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (lwmlProcessInfo_t *)malloc(sizeof(*infos)*infoCount);
            if(!infos)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*infos)*infoCount));
                return DCGM_ST_MEMORY;
            }

            lwmlReturn = lwmlDeviceGetGraphicsRunningProcesses(lwmlDevice, &infoCount, infos);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_WARNING("%d %u", "Unexpected st %d from lwmlDeviceGetGraphicsRunningProcesses on gpuId %u",
                              (int)lwmlReturn, gpuId);
                free(infos);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            for(i = 0; i<(int)infoCount; i++)
            {
                dcgmRunningProcess_t runProc = {0};

                runProc.version = dcgmRunningProcess_version;
                runProc.pid = infos[i].pid;

                if(infos[i].usedGpuMemory == (unsigned long long)LWML_VALUE_NOT_AVAILABLE)
                    runProc.memoryUsed = DCGM_INT64_NOT_SUPPORTED;
                else
                    runProc.memoryUsed = infos[i].usedGpuMemory;

                /* Append a value for each pid */
                AppendEntityBlob(threadCtx, &runProc, sizeof(runProc), now, expireTime);

                PRINT_DEBUG("%u %llu %d", "Appended graphics pid %u, usedMemory %llu to gpuId %u", runProc.pid,
                            runProc.memoryUsed, gpuId);
            }

            free(infos);
            break;
        }

        case DCGM_FI_DEV_COMPUTE_PIDS:
        {
            int i;
            unsigned int infoCount = 0;
            lwmlProcessInfo_t *infos = 0;

            /* First, get the capacity we need */
            lwmlReturn = lwmlDeviceGetComputeRunningProcesses(lwmlDevice, &infoCount, 0);
            if(lwmlReturn == LWML_SUCCESS)
            {
                PRINT_DEBUG("%d", "No compute PIDs running on gpuId %u", gpuId);
                break;
            }
            else if(lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE)
            {
                PRINT_WARNING("%d %u", "Unexpected st %d from lwmlDeviceGetComputeRunningProcesses on gpuId %u",
                              (int)lwmlReturn, gpuId);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (lwmlProcessInfo_t *)malloc(sizeof(*infos)*infoCount);
            if(!infos)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*infos)*infoCount));
                return DCGM_ST_MEMORY;
            }

            lwmlReturn = lwmlDeviceGetComputeRunningProcesses(lwmlDevice, &infoCount, infos);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_WARNING("%d %d", "Unexpected st %d from lwmlDeviceGetComputeRunningProcesses on gpuId %u",
                              (int)lwmlReturn, gpuId);
                free(infos);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            for(i = 0; i<(int)infoCount; i++)
            {
                dcgmRunningProcess_t runProc = {0};

                runProc.version = dcgmRunningProcess_version;
                runProc.pid = infos[i].pid;


                if(infos[i].usedGpuMemory == (unsigned long long)LWML_VALUE_NOT_AVAILABLE)
                    runProc.memoryUsed = DCGM_INT64_NOT_SUPPORTED;
                else
                    runProc.memoryUsed = infos[i].usedGpuMemory;

                /* Append a value for each pid */
                AppendEntityBlob(threadCtx, &runProc, sizeof(runProc), now, expireTime);

                PRINT_DEBUG("%u %llu %u", "Appended graphics pid %u, usedMemory %llu to gpuId %u", runProc.pid,
                            runProc.memoryUsed, gpuId);
            }

            free(infos);
            break;
        }
        
        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            lwmlComputeMode_t lwrrentComputeMode;

            // Get the current compute mode
            lwmlReturn = lwmlDeviceGetComputeMode(lwmlDevice, &lwrrentComputeMode);
            if (LWML_SUCCESS != lwmlReturn)
            {
                    AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0,
                                      now, expireTime);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if (lwrrentComputeMode == LWML_COMPUTEMODE_PROHIBITED)
                lwrrentComputeMode = LWML_COMPUTEMODE_EXCLUSIVE_THREAD; //Mapped to 1 since exclusive thread removed
            else if (lwrrentComputeMode == LWML_COMPUTEMODE_EXCLUSIVE_PROCESS)
                lwrrentComputeMode = LWML_COMPUTEMODE_PROHIBITED; //Mapped to 2 since Exclusive thread removed
            
            AppendEntityInt64(threadCtx, lwrrentComputeMode, 0, now, expireTime);
            break;
        }
        
        case DCGM_FI_SYNC_BOOST:
        {
            lwmlSyncBoostGroupList_t syncBoosGrptList;
            
            
            lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal,
                                        SystemGetSyncBoostGroups,
                                        (&syncBoosGrptList));
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;

            if(lwmlReturn != LWML_SUCCESS)
            {
                /* Zero out the structure. We're still going to insert it */
                memset(&syncBoosGrptList, 0, sizeof(syncBoosGrptList));
            }            
            
            AppendSyncBoostGroups(threadCtx, &syncBoosGrptList, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_ENFORCED_POWER_LIMIT:
        {
            unsigned int powerLimitInt;
            double powerLimitDbl;

            lwmlReturn = lwmlDeviceGetEnforcedPowerLimit(lwmlDevice, &powerLimitInt);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (LWML_SUCCESS != lwmlReturn) {
                AppendEntityDouble(threadCtx, LwmlErrorToDoubleValue(lwmlReturn), 0,
                                  now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            powerLimitDbl = powerLimitInt/1000;
            AppendEntityDouble(threadCtx, powerLimitDbl, 0,
                              now, expireTime);

            break;
        }
        
        case DCGM_FI_DEV_GPU_UTIL_SAMPLES:
        case DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES:
        {
            unsigned int i;
            lwmlProcessSample_t sampleBuff[100] = {{0}};
            lwmlValueType_t sampleValType;
            unsigned sampleCount = sizeof (sampleBuff) / sizeof (lwmlProcessSample_t);

            lwmlReturn = LWML_CALL_ETBL(m_etblLwmlCommonInternal, DeviceGetProcessSamples, (lwmlDevice,
                                           lastCollectionTimeStamp, &sampleValType, &sampleCount, sampleBuff));
            if (lwmlReturn != LWML_SUCCESS && lwmlReturn != LWML_ERROR_NOT_FOUND)
            {
                PRINT_ERROR("%d %u", "DeviceGetProcessSamples returned %d for gpuId %u",
                            (int)lwmlReturn, gpuId);
                return LwmlReturnToDcgmReturn(lwmlReturn);                
            }            

            if (fieldMeta->fieldId == DCGM_FI_DEV_GPU_UTIL_SAMPLES)
            {
                for (i = 0; i < sampleCount; i++) {
                    AppendEntityDouble(threadCtx, (double)sampleBuff[i].sm.util.uiVal, sampleBuff[i].sm.pid,
                            now, expireTime);
                }
            } else {
                for (i = 0; i < sampleCount; i++) {
                    AppendEntityDouble(threadCtx, (double)sampleBuff[i].mem.util.uiVal, sampleBuff[i].mem.pid,
                            now, expireTime);
                }                
            }

            break;
        }

        case DCGM_FI_DEV_ACCOUNTING_DATA:
        {
            unsigned int i;
            unsigned int maxPidCount = 0;
            unsigned int pidCount = 0;
            unsigned int *pids = 0;
            lwmlAccountingStats_t accountingStats;

            /* Find out how many PIDs we can query */
            lwmlReturn = lwmlDeviceGetAccountingBufferSize(lwmlDevice, &maxPidCount);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "lwmlDeviceGetAccountingBufferSize returned %d for gpuId %u",
                            (int)lwmlReturn, gpuId);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            /* Alloc space to hold the PID list */
            pids = (unsigned int *)malloc(sizeof(pids[0])*maxPidCount);
            if(!pids)
            {
                PRINT_ERROR("", "Malloc failure");
                return DCGM_ST_MEMORY;
            }
            memset(pids, 0, sizeof(pids[0])*maxPidCount);

            pidCount = maxPidCount;
            lwmlReturn = lwmlDeviceGetAccountingPids(lwmlDevice, &pidCount, pids);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if(lwmlReturn != LWML_SUCCESS)
            {
                PRINT_ERROR("%d %d", "lwmlDeviceGetAccountingPids returned %d for gpuId %u",
                            (int)lwmlReturn, gpuId);
                free(pids);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            PRINT_DEBUG("%u %u", "Read %u pids for gpuId %u", pidCount, gpuId);

            /* Walk over the PIDs */
            for(i = 0; i < pidCount; i++)
            {
                lwmlReturn = lwmlDeviceGetAccountingStats(lwmlDevice, pids[i], &accountingStats);
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                if(lwmlReturn != LWML_SUCCESS)
                {
                    PRINT_WARNING("%d %u %u", "lwmlDeviceGetAccountingStats returned %d for gpuId %u, pid %u",
                                  (int)lwmlReturn, (int)gpuId, pids[i]);
                    /* Keep going on more PIDs */
                    continue;
                }


                /* Append a stats record for the PID */
                AppendDeviceAccountingStats(threadCtx, pids[i], &accountingStats,
                                            now, expireTime);
            }

            free(pids);
            pids = 0;
            break;
        }

        case DCGM_FI_DEV_RETIRED_SBE:
        {
            lwmlPageRetirementCause_t cause = LWML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS;
            unsigned int pageCount = 0; /* must be 0 to retrieve count */

            lwmlReturn = lwmlDeviceGetRetiredPages(lwmlDevice, cause, &pageCount, NULL);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS && lwmlReturn != LWML_ERROR_INSUFFICIENT_SIZE)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            
            AppendEntityInt64(threadCtx, (long long)pageCount, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_PCIE_MAX_LINK_GEN:
        {
            unsigned int value = 0;
            lwmlReturn = lwmlDeviceGetMaxPcieLinkGeneration(lwmlDevice, &value);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH:
        {
            unsigned int value = 0;
            lwmlReturn = lwmlDeviceGetMaxPcieLinkWidth(lwmlDevice, &value);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_PCIE_LINK_GEN:
        {
            unsigned int value = 0;
            lwmlReturn = lwmlDeviceGetLwrrPcieLinkGeneration(lwmlDevice, &value);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_PCIE_LINK_WIDTH:
        {
            unsigned int value = 0;
            lwmlReturn = lwmlDeviceGetLwrrPcieLinkWidth(lwmlDevice, &value);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }
        
        case DCGM_FI_DEV_PSTATE:
        {
            lwmlPstates_t value = LWML_PSTATE_UNKNOWN;
            lwmlReturn = lwmlDeviceGetPerformanceState(lwmlDevice, &value);
            if(watchInfo)
                watchInfo->lastStatus = lwmlReturn;
            if (lwmlReturn != LWML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, LwmlErrorToInt64Value(lwmlReturn), 0, now, expireTime);
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_LWLINK_ERRORS:
            break; /* These are handled by the LWML event thread (m_eventThread) */

        case DCGM_FI_GPU_TOPOLOGY_AFFINITY:
        {
            dcgmReturn_t ret = CacheTopologyAffinity(threadCtx, now, expireTime);

            if (ret != DCGM_ST_OK)
                return ret;

            break;
        }
        case DCGM_FI_GPU_TOPOLOGY_LWLINK:
        {
            dcgmReturn_t ret = CacheTopologyLwLink(threadCtx, now, expireTime);

            if (ret != DCGM_ST_OK)
                return ret;
 
            break;
        }
        case DCGM_FI_GPU_TOPOLOGY_PCI:
        {
            dcgmTopology_t *topology_p;
            unsigned int elementArraySize = 0;
            unsigned int topologySize = 0;
            unsigned int elementsFilled = 0;

            unsigned int deviceCount = 0;
            lwmlReturn = lwmlDeviceGetCount(&deviceCount);
            if (LWML_SUCCESS != lwmlReturn)
            {
                PRINT_DEBUG("", "Could not retrieve device count");
                if(watchInfo)
                    watchInfo->lastStatus = lwmlReturn;
                return LwmlReturnToDcgmReturn(lwmlReturn);
            }

            if (deviceCount < 2)
            {
                PRINT_DEBUG("", "Two devices not detected on this system");
                if(watchInfo)
                    watchInfo->lastStatus = LWML_ERROR_NOT_SUPPORTED;
                return (DCGM_ST_NOT_SUPPORTED);
            }
            else if (deviceCount > DCGM_MAX_NUM_DEVICES)
            {
                PRINT_WARNING("%u", "Capping GPU topology discovery to DCGM_MAX_NUM_DEVICES even though %u were found in LWML", deviceCount);
                deviceCount = DCGM_MAX_NUM_DEVICES;
            }

            // arithmetic series formula to calc number of combinations
            elementArraySize = (unsigned int)((float)(deviceCount - 1.0) * (1.0 + ((float)deviceCount - 2.0)/2.0));

            // this is intended to minimize how much we're storing since we rarely will need all 120 entries in the element array
            topologySize = sizeof(dcgmTopology_t) - (sizeof(dcgmTopologyElement_t)*DCGM_TOPOLOGY_MAX_ELEMENTS)
                    + elementArraySize * sizeof(dcgmTopologyElement_t);

            topology_p = (dcgmTopology_t *) malloc(topologySize);

            // clear the array
            memset(topology_p, 0, topologySize);

            // topology needs to be freed for all error conditions below
            topology_p->version = dcgmTopology_version1;
            for (unsigned int index1 = 0; index1 < deviceCount; index1++)
            {
                lwmlDevice_t device1;
                lwmlReturn = lwmlDeviceGetHandleByIndex(index1, &device1);
                if (LWML_SUCCESS != lwmlReturn && LWML_ERROR_NO_PERMISSION != lwmlReturn)
                {
                    free(topology_p);
                    return LwmlReturnToDcgmReturn(lwmlReturn);
                }

                // if we cannot access this GPU then just move on
                if (LWML_ERROR_NO_PERMISSION == lwmlReturn)
                {
                    PRINT_DEBUG("%d", "Unable to access GPU %d", index1);
                    continue;
                }
    
                for (unsigned int index2 = index1+1; index2 < deviceCount; index2++)
                {
                    lwmlGpuTopologyLevel_t path;
                    lwmlDevice_t device2;
                          
                    lwmlReturn = lwmlDeviceGetHandleByIndex(index2, &device2);
                    if (LWML_SUCCESS != lwmlReturn && LWML_ERROR_NO_PERMISSION != lwmlReturn)
                    {
                        free(topology_p);
                        return LwmlReturnToDcgmReturn(lwmlReturn);
                    }

                    // if we cannot access this GPU then just move on
                    if (LWML_ERROR_NO_PERMISSION == lwmlReturn)
                    {
                        PRINT_DEBUG("%d", "Unable to access GPU %d", index2);
                        continue;
                    }

                    lwmlReturn = lwmlDeviceGetTopologyCommonAncestor(device1, device2, &path);
                    if (LWML_SUCCESS != lwmlReturn)
                    {
                        free(topology_p);
                        return LwmlReturnToDcgmReturn(lwmlReturn);
                    }
                        
                    topology_p->element[elementsFilled].dcgmGpuA = LwmlIndexToGpuId(index1);
                    topology_p->element[elementsFilled].dcgmGpuB = LwmlIndexToGpuId(index2);
                    switch (path)
                    {
                        case LWML_TOPOLOGY_INTERNAL:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_BOARD;
                            break;
                        case LWML_TOPOLOGY_SINGLE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_SINGLE;
                            break;
                        case LWML_TOPOLOGY_MULTIPLE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_SINGLE;
                            break;
                        case LWML_TOPOLOGY_HOSTBRIDGE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_HOSTBRIDGE;
                            break;
                        case LWML_TOPOLOGY_CPU:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_CPU;
                            break;
                        case LWML_TOPOLOGY_SYSTEM:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_SYSTEM;
                            break;
                        default:
                            free(topology_p);
                            PRINT_ERROR("", "Received an invalid value as a path from the common ancestor call");
                            return DCGM_ST_GENERIC_ERROR;
                    }
                    elementsFilled++;
                }
            }
            topology_p->numElements = elementsFilled;
 
            AppendEntityBlob(threadCtx, topology_p, topologySize, now, expireTime);
            free(topology_p);
            break;
        }

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ManageVgpuList(unsigned int gpuId, lwmlVgpuInstance_t *vgpuInstanceIds)
{
    dcgmcm_vgpu_info_p lwrr = NULL, temp = NULL, initialVgpuListState = NULL, finalVgpuListState = NULL;

    /* First element of the vgpuInstanceIds array must hold the count of vGPU instances running */
    unsigned int vgpuCount = vgpuInstanceIds[0];

    /* Stores the initial state of vgpuList for the current GPU */
    initialVgpuListState = m_gpus[gpuId].vgpuList;

    /* Checking if new vGPU instances have spawned up in this iteration.
       A vGPU instance is new if its instance ID is present in the refreshed
       list returned by LWML, but is absent from the current GPU's vgpuList. */
    for (unsigned int i = 0; i < vgpuCount; i++)
    {
        bool new_entry = 1;
        temp = lwrr = m_gpus[gpuId].vgpuList;

        while (lwrr)
        {
            if (vgpuInstanceIds[i+1] == lwrr->vgpuId)
            {
                new_entry = 0;
                /* Marking the current vgpuList entry as "found", since it is present in
                   the refreshed vGPU ID list returned by LWML */
                lwrr->found = 1;
                break;
            }
            temp = lwrr;
            lwrr = lwrr->next;
        }

        /* Add the new vGPU Instance info to the vgpuList of the current GPU */
        if (new_entry)
        {
            int lwrrVgpuIdx = 0;
            int startIdx    = LWCMCM_START_VGPU_IDX_FOR_GPU(gpuId);
            int endIdx      = LWCMCM_END_VGPU_IDX_FOR_GPU(gpuId);

            dcgmcm_vgpu_info_p vgpuInfo = (dcgmcm_vgpu_info_p)malloc(sizeof(dcgmcm_vgpu_info_t));
            if(!vgpuInfo)
            {
                PRINT_ERROR("%d %u", "malloc of %d bytes failed for metadata of vGPU instance %u",
                            (int)(sizeof(dcgmcm_vgpu_info_t)), vgpuInstanceIds[i+1]);
                continue;
            }

            vgpuInfo->vgpuId    = vgpuInstanceIds[i+1];
            vgpuInfo->found     = 1;
            vgpuInfo->next      = NULL;


            dcgm_mutex_lock(m_mutex);
            if (!m_gpus[gpuId].vgpuList)
                m_gpus[gpuId].vgpuList = vgpuInfo;
            else
            {
                vgpuInfo->next = temp->next;
                temp->next = vgpuInfo;
            }
            dcgm_mutex_unlock(m_mutex);
        
            WatchVgpuFields(vgpuInfo->vgpuId);
        }
    }

    temp = lwrr = m_gpus[gpuId].vgpuList;

    /* Remove entries of inactive vGPU instances from the current GPU's list. */
    while (lwrr)
    {
        dcgmcm_vgpu_info_p toBeDeleted = NULL;
        /*Any vGPU metadata node in m_vgpus that is not marked as "found" in the previous loop is stale/inactive. */
        if (!lwrr->found)
        {
            dcgm_mutex_lock(m_mutex);
            toBeDeleted = lwrr;
            if (lwrr == m_gpus[gpuId].vgpuList)
                m_gpus[gpuId].vgpuList = lwrr->next;
            else
                temp->next = lwrr->next;
            lwrr = lwrr->next;
            dcgm_mutex_unlock(m_mutex);

            UnwatchVgpuFields(toBeDeleted->vgpuId);
            PRINT_DEBUG("%u %u", "Removing vgpuId %u for gpuId %u", toBeDeleted->vgpuId, gpuId);
            free(toBeDeleted);
        }
        else
        {
            dcgm_mutex_lock(m_mutex);
            lwrr->found = 0;
            temp = lwrr;
            lwrr = lwrr->next;
            dcgm_mutex_unlock(m_mutex);
        }
    }

    /* Stores the final state of vgpuList after any addition/removal of vGPU entries on the current GPU */
    finalVgpuListState = m_gpus[gpuId].vgpuList;

    DcgmWatcher watcher(DcgmWatcherTypeCacheManager);

    /* Watching frequently cached fields only when there are vGPU instances running on the GPU. */
    if((!initialVgpuListState) && (finalVgpuListState))
    {
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_UTILIZATIONS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ENC_STATS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_STATS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_SESSIONS_INFO, 1000000, 600.0, 600, watcher, false);
    }
    else if((initialVgpuListState) && (!finalVgpuListState))
    {
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_UTILIZATIONS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ENC_STATS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_STATS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_SESSIONS_INFO, 1, watcher);
    }

    /* Verifying vpuList to match the input vGPU instance ids array, in case of mismatch return DCGM_ST_GENERIC_ERROR */
    temp = m_gpus[gpuId].vgpuList;
    while(temp)
    {
        unsigned int i = 0;
        while (i < vgpuCount && temp->vgpuId != vgpuInstanceIds[i+1])
        {
            i++;
        }
        if (i >= vgpuCount)
            return DCGM_ST_GENERIC_ERROR;
        temp = temp->next;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetDeviceFBCSessionsInfo(lwmlDevice_t lwmlDevice, dcgmcm_update_thread_t *threadCtx,
                                                        dcgmcm_watch_info_p watchInfo, timelib64_t now, timelib64_t expireTime)
{
    dcgmDeviceFbcSessions_t *devFbcSessions = NULL;
    lwmlFBCSessionInfo_t *sessionInfo = NULL;
    unsigned int i, sessionCount = 0;
    lwmlReturn_t lwmlReturn;

    devFbcSessions = (dcgmDeviceFbcSessions_t *)malloc(sizeof(*devFbcSessions));
    if(!devFbcSessions)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*devFbcSessions)));
        return DCGM_ST_MEMORY;
    }

    lwmlReturn = lwmlDeviceGetFBCSessions(lwmlDevice, &sessionCount, NULL);
    if(watchInfo)
        watchInfo->lastStatus = lwmlReturn;

    if (lwmlReturn != LWML_SUCCESS || sessionCount == 0)
    {
        devFbcSessions->version = dcgmDeviceFbcSessions_version;
        devFbcSessions->sessionCount = 0;
        int payloadSize = sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo);
        AppendEntityBlob(threadCtx, devFbcSessions, payloadSize,
                         now, expireTime);
        free(devFbcSessions);
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }

    sessionInfo = (lwmlFBCSessionInfo_t *)malloc(sizeof(*sessionInfo)*(sessionCount));
    if(!sessionInfo)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo)*(sessionCount)));
        free(devFbcSessions);
        return DCGM_ST_MEMORY;
    }

    lwmlReturn = lwmlDeviceGetFBCSessions(lwmlDevice, &sessionCount, sessionInfo);
    if(watchInfo)
        watchInfo->lastStatus = lwmlReturn;
    if (lwmlReturn != LWML_SUCCESS)
    {
        PRINT_ERROR("%d", "lwmlDeviceGetFBCSessions failed with status %d",
                      (int)lwmlReturn);
        free(sessionInfo);
        free(devFbcSessions);
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }

    devFbcSessions->version = dcgmDeviceFbcSessions_version;
    devFbcSessions->sessionCount = sessionCount;

    for (i = 0; i < sessionCount; i++)
    {
        if(devFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break; /* Don't overflow data structure */

        devFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        devFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        devFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        devFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        devFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        devFbcSessions->sessionInfo[i].sessionType    = (dcgmFBCSessionType_t)sessionInfo[i].sessionType;
        devFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        devFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        devFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        devFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        devFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        devFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        devFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    /* Only store as much as is actually populated */
    int payloadSize = (sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo)) +
                      (devFbcSessions->sessionCount * sizeof(devFbcSessions->sessionInfo[0]));

    AppendEntityBlob(threadCtx, devFbcSessions, payloadSize,
                     now, expireTime);
    free(sessionInfo);
    free(devFbcSessions);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetVgpuInstanceFBCSessionsInfo(lwmlVgpuInstance_t vgpuId, dcgmcm_update_thread_t *threadCtx,
                                                              dcgmcm_watch_info_p watchInfo, timelib64_t now, timelib64_t expireTime)
{
    dcgmDeviceFbcSessions_t *vgpuFbcSessions = NULL;
    lwmlFBCSessionInfo_t *sessionInfo = NULL;
    unsigned int i, sessionCount = 0;
    lwmlReturn_t lwmlReturn;

    vgpuFbcSessions = (dcgmDeviceFbcSessions_t *)malloc(sizeof(*vgpuFbcSessions));
    if(!vgpuFbcSessions)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuFbcSessions)));
        return DCGM_ST_MEMORY;
    }

    lwmlReturn = lwmlVgpuInstanceGetFBCSessions(vgpuId, &sessionCount, NULL);
    if(watchInfo)
        watchInfo->lastStatus = lwmlReturn;

    if (lwmlReturn != LWML_SUCCESS || sessionCount == 0)
    {
        vgpuFbcSessions->version = dcgmDeviceFbcSessions_version;
        vgpuFbcSessions->sessionCount = 0;
        int payloadSize = sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo);
        AppendEntityBlob(threadCtx, vgpuFbcSessions, payloadSize,
                         now, expireTime);
        free(vgpuFbcSessions);
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }

    sessionInfo = (lwmlFBCSessionInfo_t *)malloc(sizeof(*sessionInfo)*(sessionCount));
    if(!sessionInfo)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo)*(sessionCount)));
        free(vgpuFbcSessions);
        return DCGM_ST_MEMORY;
    }

    lwmlReturn = lwmlVgpuInstanceGetFBCSessions(vgpuId, &sessionCount, sessionInfo);
    if(watchInfo)
        watchInfo->lastStatus = lwmlReturn;
    if (lwmlReturn != LWML_SUCCESS)
    {
        PRINT_ERROR("%d %u", "lwmlVgpuInstanceGetFBCSessions failed with status %d for vgpuId %u",
                      (int)lwmlReturn, vgpuId);
        free(sessionInfo);
        free(vgpuFbcSessions);
        return LwmlReturnToDcgmReturn(lwmlReturn);
    }

    vgpuFbcSessions->version = dcgmDeviceFbcSessions_version;
    vgpuFbcSessions->sessionCount = sessionCount;

    for (i = 0; i < sessionCount; i++)
    {
        if(vgpuFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break; /* Don't overflow data structure */

        vgpuFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        vgpuFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        vgpuFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        vgpuFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        vgpuFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        vgpuFbcSessions->sessionInfo[i].sessionType    = (dcgmFBCSessionType_t)sessionInfo[i].sessionType;
        vgpuFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        vgpuFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        vgpuFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        vgpuFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        vgpuFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        vgpuFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        vgpuFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    /* Only store as much as is actually populated */
    int payloadSize = (sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo)) +
                      (vgpuFbcSessions->sessionCount * sizeof(vgpuFbcSessions->sessionInfo[0]));

    AppendEntityBlob(threadCtx, vgpuFbcSessions, payloadSize,
                     now, expireTime);
    free(sessionInfo);
    free(vgpuFbcSessions);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceAccountingStats(dcgmcm_update_thread_t *threadCtx,
                                                           unsigned int pid,
                                                           lwmlAccountingStats_t *lwmlAccountingStats,
                                                           timelib64_t timestamp, timelib64_t oldestKeepTimestamp)
{
    dcgmDevicePidAccountingStats_t accountingStats;

    memset(&accountingStats, 0, sizeof(accountingStats));
    accountingStats.version = dcgmDevicePidAccountingStats_version;

    accountingStats.pid = pid;
    accountingStats.gpuUtilization = lwmlAccountingStats->gpuUtilization;
    accountingStats.memoryUtilization = lwmlAccountingStats->memoryUtilization;
    accountingStats.maxMemoryUsage = lwmlAccountingStats->maxMemoryUsage;
    accountingStats.startTimestamp = lwmlAccountingStats->startTime;
    accountingStats.activeTimeUsec = lwmlAccountingStats->time * 1000;

    dcgm_mutex_lock(m_mutex);

    /* Use startTimestamp as the 2nd key since that won't change */
    if(HasAccountingPidBeenSeen(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp))
    {
        dcgm_mutex_unlock(m_mutex);
        PRINT_DEBUG("%u %llu", "Skipping pid %u, startTimestamp %llu that has already been seen",
                    accountingStats.pid, accountingStats.startTimestamp);
        return DCGM_ST_OK;
    }

    /* Cache the PID when the process completes as no further updates will be required for the process */
    if (accountingStats.activeTimeUsec > 0) {
        CacheAccountingPid(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp);
    }

    dcgm_mutex_unlock(m_mutex);

    AppendEntityBlob(threadCtx, &accountingStats, sizeof(accountingStats), timestamp, oldestKeepTimestamp);

    PRINT_DEBUG("%u %u %u %llu %llu %llu", "Recording PID %u, gpu %u, mem %u, maxMemory %llu, startTs %llu, activeTime %llu",
                accountingStats.pid, accountingStats.gpuUtilization, accountingStats.memoryUtilization,
                accountingStats.maxMemoryUsage, accountingStats.startTimestamp, accountingStats.activeTimeUsec);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceSupportedClocks(dcgmcm_update_thread_t *threadCtx,
                                             lwmlSupportedClocks_t *supportedClocks,
                                             timelib64_t timestamp, timelib64_t oldestKeepTimestamp)
{
    dcgmDeviceSupportedClockSets_t supClockSets;
    int memClockIdx, grClockIdx;
    unsigned short memClk;

    memset(&supClockSets, 0, sizeof(supClockSets));
    supClockSets.version = dcgmDeviceSupportedClockSets_version;

    for(memClockIdx = 0; memClockIdx < (int)supportedClocks->memoryClocksCount; memClockIdx++)
    {
        memClk = supportedClocks->memoryClocks[memClockIdx];

        for(grClockIdx = 0; grClockIdx < (int)supportedClocks->graphicsClocksCount[memClockIdx]; grClockIdx++)
        {
            if(supClockSets.count >= DCGM_MAX_CLOCKS)
                break; /* Don't overflow data structure */

            supClockSets.clockSet[supClockSets.count].version = dcgmClockSet_version;
            supClockSets.clockSet[supClockSets.count].memClock = (unsigned short)memClk;
            supClockSets.clockSet[supClockSets.count].smClock = supportedClocks->graphicsClocks[memClockIdx][grClockIdx];
            supClockSets.count++;
        }
    }

    /* Only store as much as is actually populated */
    int payloadSize = (sizeof(supClockSets) - sizeof(supClockSets.clockSet)) +
                      (supClockSets.count * sizeof(supClockSets.clockSet[0]));

    AppendEntityBlob(threadCtx, &supClockSets, payloadSize, timestamp, oldestKeepTimestamp);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendSyncBoostGroups(dcgmcm_update_thread_t *threadCtx,
                                    lwmlSyncBoostGroupList_t* pLwmlSyncBoostList, timelib64_t timestamp,
                                    timelib64_t oldestKeepTimestamp)
{
    dcgmSyncBoostGroupList_t stDcgmSyncBoostList;
    
    memset(&stDcgmSyncBoostList, 0, sizeof(stDcgmSyncBoostList));
    stDcgmSyncBoostList.version = dcgmSyncBoostGroupList_version;
    
    stDcgmSyncBoostList.numGroups = pLwmlSyncBoostList->numGroups;
    
    for (int i = 0; i < pLwmlSyncBoostList->numGroups; i++) 
    {
        lwmlSyncBoostGroupListItem_t *pSyncBoostGrp;
        
        pSyncBoostGrp = &pLwmlSyncBoostList->groups[i];
        
        stDcgmSyncBoostList.syncBoostGroups[i].numDevices = pSyncBoostGrp->numDevices;
        stDcgmSyncBoostList.syncBoostGroups[i].rmGroupId = pSyncBoostGrp->groupId;
        
        for (int j = 0; j < pSyncBoostGrp->numDevices; j++) {
            unsigned int lwmlIndex;
            lwmlReturn_t retLwml;
            
            retLwml = lwmlDeviceGetIndex(pSyncBoostGrp->devices[j], &lwmlIndex);
            if (LWML_SUCCESS != retLwml) {
                
            }
            
            stDcgmSyncBoostList.syncBoostGroups[i].lwmlIndex[j] = lwmlIndex;
        }
    }
    
    AppendEntityBlob(threadCtx, &stDcgmSyncBoostList, sizeof(stDcgmSyncBoostList), timestamp,
                     oldestKeepTimestamp);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::LwmlReturnToDcgmReturn(lwmlReturn_t lwmlReturn)
{
    switch(lwmlReturn)
    {
        case LWML_SUCCESS:
            return DCGM_ST_OK;

        case LWML_ERROR_NOT_SUPPORTED:
            return DCGM_ST_NOT_SUPPORTED;

        case LWML_ERROR_NO_PERMISSION:
            return DCGM_ST_NO_PERMISSION;

        case LWML_ERROR_NOT_FOUND:
            return DCGM_ST_NO_DATA;

        case LWML_ERROR_TIMEOUT:
            return DCGM_ST_TIMEOUT;

        case LWML_ERROR_GPU_IS_LOST:
            return DCGM_ST_GPU_IS_LOST;

        case LWML_ERROR_RESET_REQUIRED:
            return DCGM_ST_RESET_REQUIRED;
            
        case LWML_ERROR_ILWALID_ARGUMENT:
            return DCGM_ST_BADPARAM;

        default:
        case LWML_ERROR_IRQ_ISSUE:
        case LWML_ERROR_LIBRARY_NOT_FOUND:
        case LWML_ERROR_FUNCTION_NOT_FOUND:
        case LWML_ERROR_CORRUPTED_INFOROM:
        case LWML_ERROR_OPERATING_SYSTEM:
        case LWML_ERROR_LIB_RM_VERSION_MISMATCH:
        case LWML_ERROR_ALREADY_INITIALIZED:
        case LWML_ERROR_UNINITIALIZED:
        case LWML_ERROR_UNKNOWN:
        case LWML_ERROR_INSUFFICIENT_SIZE:
        case LWML_ERROR_INSUFFICIENT_POWER:
        case LWML_ERROR_DRIVER_NOT_LOADED:
            return DCGM_ST_LWML_ERROR;
    }

    return DCGM_ST_GENERIC_ERROR; /* Shouldn't get here */
}

int DcgmCacheManager::LwmlGpuLWLinkErrorToDcgmError(long eventType)
{
    switch(eventType)
    {
        case lwmlEventTypeLWLinkRecoveryError:
            return DCGM_GPU_LWLINK_ERROR_RECOVERY_REQUIRED;
        case lwmlEventTypeLWLinkFatalError:
            return DCGM_GPU_LWLINK_ERROR_FATAL;
    }

    return 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GpuReset(unsigned int gpuId)
{
    // this taken almost verbatim from LWSMI's reset GPU

    lwmlEnableState_t persistenceMode;
    dcgmReturn_t retSt = DCGM_ST_OK;
    lwmlReturn_t lwmlResult = LWML_SUCCESS;
    dcgmcm_gpu_info_t *gpuInfo; /* Cached pointer to the GPU we are working on */

    PRINT_DEBUG("%d", "Request to reset GPU ID %d", gpuId);

    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    /* We don't want two modules in here. Lock for the duration of this function */
    dcgm_mutex_lock(m_mutex);

    gpuInfo = &m_gpus[gpuId];

    //Trigger the reset if the gpu status is ok or disabled
    if(gpuInfo->status != DcgmcmGpuStatusOk && gpuInfo->status != DcgmcmGpuStatusDisabled)
    {
        if(gpuInfo->status == DcgmcmGpuStatusUnsupported)
            retSt = DCGM_ST_GPU_NOT_SUPPORTED;
        else
            retSt = DCGM_ST_GENERIC_ERROR;

        dcgm_mutex_unlock(m_mutex);
        PRINT_WARNING("%u %d", "Skipping reset of gpuId %u in state %d", gpuId, gpuInfo->status);
        return retSt;
    }

    /* Make sure the GPU LwLink states are up to date before we use them */
    UpdateLwLinkLinkState(gpuId);

    /* Bug 200377294: We can't reset the GPU if any LwLinks are active */
    bool lwlinksAreActive = false;
    int i;
    for(i = 0; i < DCGM_LWLINK_MAX_LINKS_PER_GPU; i++)
    {
        if(gpuInfo->lwLinkLinkState[i] == DcgmLwLinkLinkStateUp)
        {
            PRINT_WARNING("%u %d", "Skipping GPU reset for gpuId %u because LwLink LinkId %d is active",
                          gpuId, i);
            lwlinksAreActive = true;
            break;
        }
    }

    if(lwlinksAreActive)
    {
        dcgm_mutex_unlock(m_mutex);
        /* Warning is printed above */
        return DCGM_ST_NOT_SUPPORTED;
    }


    /* Prevent any updates to this GPU while we reset it */
    PauseGpu(gpuId);

    /* See if there's a pending GOM change. If there is, we can't do a reset */
    {
        lwmlGpuOperationMode_t lwrrentGom, pendingGom;
        lwmlResult = lwmlDeviceGetGpuOperationMode(gpuInfo->lwmlDevice, &lwrrentGom, &pendingGom);
        if (LWML_SUCCESS == lwmlResult)
        {
            if (lwrrentGom != pendingGom)
            {
                PRINT_ERROR("%u %d", "For gpuId %u, lwmlIndex %d GPU Reset couldn't run because there is a GPU"
                            " Operation Mode change in flight. Please reboot your system.",
                            gpuId, gpuInfo->lwmlIndex);
                ResumeGpu(gpuId);
                dcgm_mutex_unlock(m_mutex);
                return LwmlReturnToDcgmReturn(lwmlResult);
            }
        }
        else if (LWML_ERROR_NOT_SUPPORTED != lwmlResult)
        {
            // TODO Can't reboot a lost GPU via the SW mechanism
            PRINT_ERROR("%u %d", "For gpuId %u, lwmlIndex %d, GPU Reset couldn't run due to problem with internal"
                        " state of the GPU. Please reboot your system.",
                        gpuId, gpuInfo->lwmlIndex);
            ResumeGpu(gpuId);
            dcgm_mutex_unlock(m_mutex);
            return LwmlReturnToDcgmReturn(lwmlResult);
        }
    }

    // Bug 879812: We can't perform a GPU reset in persistence mode. Hence, if we are in persistence mode,
    // we temporarily disable it, perform the reset and then reenable it
    lwmlResult = lwmlDeviceGetPersistenceMode(gpuInfo->lwmlDevice, &persistenceMode);
    if (LWML_SUCCESS != lwmlResult)
    {
        PRINT_ERROR("%u %d %s", "For gpuId %u, lwmlIndex %d, Failed to get persistence mode. lwml error: %s",
                    gpuId, gpuInfo->lwmlIndex, lwmlErrorString(lwmlResult));
        ResumeGpu(gpuId);
        dcgm_mutex_unlock(m_mutex);
        return LwmlReturnToDcgmReturn(lwmlResult);
    }

    if(LWML_FEATURE_ENABLED == persistenceMode)
    {
        lwmlResult = lwmlDeviceSetPersistenceMode(gpuInfo->lwmlDevice, LWML_FEATURE_DISABLED);
        if(LWML_SUCCESS != lwmlResult)
        {
            PRINT_ERROR("%d", "lwmlDeviceSetPersistenceMode failed with %d", lwmlResult);
            ResumeGpu(gpuId);
            dcgm_mutex_unlock(m_mutex);
            return LwmlReturnToDcgmReturn(lwmlResult);
        }
    }

    lwmlResult = LWML_CALL_ETBL(m_etblLwmlCommonInternal, DeviceReset, (gpuInfo->lwmlDevice));
    if (lwmlResult != LWML_SUCCESS)
    {
        PRINT_ERROR("%u %d", "Reset GPU ID %u, LWML Device Reset call failed with: %d",
                    gpuId, lwmlResult);

        /* Not safe to enable the GPUs when GPU is lost or when there is a unknown error */
        if (LWML_ERROR_GPU_IS_LOST == lwmlResult)
        {
            PRINT_WARNING("%u", "gpuId %u is lost", gpuId);
            gpuInfo->status = DcgmcmGpuStatusGpuLost;
            retSt = DCGM_ST_GPU_IS_LOST; /* Redundant. Keep in case we refactor the return below */
        }
    }

    lwmlReturn_t lwmlResultReattach = LWML_SUCCESS;

    // Even if the GPU reset failed, try to restore the persistence mode!
    if (LWML_FEATURE_ENABLED == persistenceMode)
    {
        // Note: We need to teardown the entire API in order for all the device-handles
        // to close and the reset to be properly applied on the next lwmlInit()

        // Re-select the device since the handle is no longer valid!
        if (LWML_SUCCESS != (lwmlResultReattach = lwmlShutdown()) ||
           LWML_SUCCESS != (lwmlResultReattach = lwmlInit()) ||
           LWML_SUCCESS != (lwmlResultReattach = lwmlDeviceGetHandleByPciBusId(gpuInfo->pciInfo.busId, &gpuInfo->lwmlDevice)) ||
           LWML_SUCCESS != (lwmlResultReattach = lwmlDeviceSetPersistenceMode(gpuInfo->lwmlDevice, LWML_FEATURE_ENABLED)))
        {
            lwmlResult = lwmlResultReattach;
        }
    }
    else
    {
        /* WAR till the time we get per GPU reset working */
        if(LWML_SUCCESS != (lwmlResultReattach = lwmlShutdown()) ||
           LWML_SUCCESS != (lwmlResultReattach = lwmlInit()) ||
           LWML_SUCCESS != (lwmlResultReattach = lwmlDeviceGetHandleByPciBusId(gpuInfo->pciInfo.busId, &gpuInfo->lwmlDevice)))
        {
            lwmlResult = lwmlResultReattach;
        }
    }

    if (LWML_SUCCESS != lwmlResultReattach)
    {
        PRINT_ERROR("%u %d", "Reset GPU ID %u, reselect device handle failed : %d. Marking as LOST", 
                    gpuId, lwmlResultReattach);
        gpuInfo->status = DcgmcmGpuStatusGpuLost;
        ResumeGpu(gpuId);
        dcgm_mutex_unlock(m_mutex);
        return LwmlReturnToDcgmReturn(lwmlResultReattach);
    }

    /* Force ECC values to re-read since they may have changed as a result of the reset */
    dcgmcm_watch_info_p eccLwrrentWatchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_LWRRENT, 1);
    dcgmcm_watch_info_p eccPendingWatchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_PENDING, 1);

    if(eccLwrrentWatchInfo->isWatched)
    {
        /* Force field to refresh */
        eccLwrrentWatchInfo->lastQueriedUsec = 0;
    }
    if(eccPendingWatchInfo->isWatched)
    {
        /* Force field to refresh */
        eccPendingWatchInfo->lastQueriedUsec = 0;
    }

    ResumeGpu(gpuId);

    dcgm_mutex_unlock(m_mutex);

    /* Wait for the fields to update if they were watched */
    if(eccLwrrentWatchInfo->isWatched || eccPendingWatchInfo->isWatched)
    {
        PRINT_DEBUG("", "Reset GPU waiting for field update");
        UpdateAllFields(1);
    }

    PRINT_DEBUG("%d", "Reset GPU ID %d successful", gpuId);
    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds)
{
    gpuIds.clear();

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if(!activeOnly || m_gpus[i].status == DcgmcmGpuStatusOk || 
           m_gpus[i].status == DcgmcmGpuStatusFakeGpu)
        {
            gpuIds.push_back(m_gpus[i].gpuId);
        }
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmCacheManager::GetGpuCount(int activeOnly)
{
    int count = 0;

    if(!activeOnly)
        return m_numGpus; /* Easy answer */

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if(m_gpus[i].status == DcgmcmGpuStatusOk || 
           m_gpus[i].status == DcgmcmGpuStatusFakeGpu)
        {
            count++;
        }
    }

    dcgm_mutex_unlock(m_mutex);

    return count;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetAllEntitiesOfEntityGroup(int activeOnly, 
                                     dcgm_field_entity_group_t entityGroupId, 
                                     std::vector<dcgmGroupEntityPair_t> &entities)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmGroupEntityPair_t insertPair;

    entities.clear();
    insertPair.entityGroupId = entityGroupId;

    dcgm_mutex_lock(m_mutex);

    switch(entityGroupId)
    {
        case DCGM_FE_GPU:
            for (unsigned int i = 0; i < m_numGpus; i++)
            {
                if (m_gpus[i].status == DcgmcmGpuStatusDetached)
                    continue;

                if(!activeOnly || m_gpus[i].status == DcgmcmGpuStatusOk || 
                   m_gpus[i].status == DcgmcmGpuStatusFakeGpu)
                {
                    insertPair.entityId = m_gpus[i].gpuId;
                    entities.push_back(insertPair);
                }
            }
            break;

        case DCGM_FE_SWITCH:
            {
                for (int i = 0; i < m_numLwSwitches; i++)
                {
                    if(!activeOnly || m_lwSwitches[i].status == DcgmcmGpuStatusOk || 
                       m_lwSwitches[i].status == DcgmcmGpuStatusFakeGpu)
                    {
                        insertPair.entityId = m_lwSwitches[i].physicalId;
                        entities.push_back(insertPair);
                    }
                }
            }
            break;
        
        default:
        case DCGM_FE_VGPU:
        case DCGM_FE_NONE:
            PRINT_DEBUG("%u", "GetAllEntitiesOfEntityGroup entityGroupId %u not supported", 
                        entityGroupId);
            retSt = DCGM_ST_NOT_SUPPORTED;
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    return retSt;
}

/*****************************************************************************/
DcgmcmGpuStatus_t DcgmCacheManager::GetEntityStatus(dcgm_field_entity_group_t entityGroupId, 
                                                    dcgm_field_eid_t entityId)
{
    int i;
    dcgmReturn_t retSt = DCGM_ST_OK;
    DcgmcmGpuStatus_t entityStatus = DcgmcmGpuStatusUnknown;

    dcgm_mutex_lock(m_mutex);

    switch(entityGroupId)
    {
        case DCGM_FE_GPU:
            if (entityId < 0 || entityId >= m_numGpus)
                break; /* Not a valid GPU */
            
            entityStatus = m_gpus[entityId].status;
            break;

        case DCGM_FE_SWITCH:
            {
                for(i = 0; i < m_numLwSwitches; i++)
                {
                    if(m_lwSwitches[i].physicalId == entityId)
                    {
                        entityStatus = m_lwSwitches[i].status;
                        break;
                    }
                }
            }
            break;
        
        default:
        case DCGM_FE_VGPU:
        case DCGM_FE_NONE:
            PRINT_DEBUG("%u", "GetEntityStatus entityGroupId %u not supported", 
                        entityGroupId);
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    return entityStatus;
}

/*****************************************************************************/
int DcgmCacheManager::AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds)
{
    unsigned int gpuId;
    std::vector<unsigned int>::iterator gpuIt;
    dcgmcm_gpu_info_p firstGpuInfo = 0;
    dcgmcm_gpu_info_p gpuInfo = 0;

    if((int)gpuIds.size() < 2)
    {
        PRINT_DEBUG("%d", "All GPUs in list of %d are the same", (int)gpuIds.size());
        return 1;
    }

    for(gpuIt = gpuIds.begin(); gpuIt != gpuIds.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        if (gpuId >= m_numGpus)
        {
            PRINT_ERROR("%u", "Invalid gpuId %u passed to AreAllGpuIdsSameSku()", gpuId);
            return 0;
        }

        gpuInfo = &m_gpus[gpuId];
        /* Have we seen a GPU yet? If not, cache the first one we see. That will
         * be the baseline to compare against
         */
        if(!firstGpuInfo)
        {
            firstGpuInfo = gpuInfo;
            continue;
        }

        if(gpuInfo->pciInfo.pciDeviceId != firstGpuInfo->pciInfo.pciDeviceId ||
           gpuInfo->pciInfo.pciSubSystemId != firstGpuInfo->pciInfo.pciSubSystemId)
        {
            PRINT_DEBUG("%u %X %X %u %X %X", "gpuId %u pciDeviceId %X or SSID %X does not "
                        "match gpuId %u pciDeviceId %X SSID %X",
                        gpuInfo->gpuId, gpuInfo->pciInfo.pciDeviceId, gpuInfo->pciInfo.pciSubSystemId,
                        firstGpuInfo->gpuId, firstGpuInfo->pciInfo.pciDeviceId, firstGpuInfo->pciInfo.pciSubSystemId);
            return 0;
        }
    }

    PRINT_DEBUG("%d", "All GPUs in list of %d are the same", (int)gpuIds.size());
    return 1;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuFieldBytesUsed(unsigned int gpuId, unsigned short dcgmFieldId, long long *bytesUsed)
{
    dcgmReturn_t status = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
    {
        PRINT_ERROR("%u", "could not find field ID %u", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    // ensure that checking if a field is watched and then retrieving its bytes used is atomic
    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if (!watchInfo || !watchInfo->isWatched)
    {
        PRINT_ERROR("%u %u", "trying to get approximate bytes used to store a field that is not watched.  Field ID: %u, gpu ID: %u",
                    dcgmFieldId, gpuId);
        status = DCGM_ST_NOT_WATCHED;
    }
    else if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        PRINT_ERROR("%d %u %d", "field ID must have DEVICE scope (%d). field ID: %u, scope: %d",
                    DCGM_FS_DEVICE, dcgmFieldId, fieldMeta->scope);
        status = DCGM_ST_BADPARAM;
    }
    else
    {
        if(watchInfo->timeSeries)
            (*bytesUsed) += timeseries_bytes_used(watchInfo->timeSeries);
    }

    dcgm_mutex_unlock(m_mutex);

    if (DCGM_ST_OK != status)
        return status;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGlobalFieldBytesUsed(unsigned short dcgmFieldId, long long *bytesUsed)
{
    int ret;
    dcgmReturn_t status = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    if (!bytesUsed) {
        PRINT_ERROR("", "bytesUsed cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if(!fieldMeta)
    {
        PRINT_ERROR("%u", "could not find field ID %u", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    // ensure that checking if a field is watched and then retrieving its bytes used is atomic
    dcgm_mutex_lock(m_mutex);

    watchInfo = this->GetGlobalWatchInfo(dcgmFieldId, 0);
    if(!watchInfo || !watchInfo->isWatched)
    {
        PRINT_ERROR("%u", "trying to get approximate bytes used to store a field that is not watched.  Field ID: %u",
                    dcgmFieldId);
        status = DCGM_ST_NOT_WATCHED;
    }
    else if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        PRINT_ERROR("%d %u %d", "field ID must have GLOBAL scope (%u). field ID: %u, scope: %d",
                DCGM_FS_GLOBAL, dcgmFieldId, fieldMeta->scope);
        status = DCGM_ST_BADPARAM;
    }
    else
    {
        if(watchInfo->timeSeries)
            (*bytesUsed) += timeseries_bytes_used(watchInfo->timeSeries);
    }

    dcgm_mutex_unlock(m_mutex);

    if (DCGM_ST_OK != status)
        return status;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CheckValidGlobalField(unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "dcgmFieldId is invalid: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        PRINT_ERROR("%u", "field %u does not have scope DCGM_FS_GLOBAL", dcgmFieldId);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CheckValidGpuField(unsigned int gpuId, unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        PRINT_ERROR("%u", "field %u does not have scope DCGM_FS_DEVICE", dcgmFieldId);
        return DCGM_ST_BADPARAM;
    }

    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u", "invalid gpuId: %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGlobalFieldExecTimeUsec(unsigned short dcgmFieldId, long long *totalUsec)
{
    dcgmcm_watch_info_p watchInfo;
    
    if (!totalUsec)
    {
        PRINT_ERROR("", "totalUsec cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGlobalField(dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;
    
    *totalUsec = 0;
    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);
    if(watchInfo)
        *totalUsec = (long long)watchInfo->execTimeUsec;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGpuFieldExecTimeUsec(unsigned int gpuId,
                                                       unsigned short dcgmFieldId,
                                                       long long *totalUsec)
{
    dcgmcm_watch_info_p watchInfo;

    if (!totalUsec)
    {
        PRINT_ERROR("", "totalUsec cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGpuField(gpuId, dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *totalUsec = 0;
    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if(watchInfo)
    {
        *totalUsec = (long long)watchInfo->execTimeUsec;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGlobalFieldFetchCount(unsigned short dcgmFieldId,
                                                        long long *fetchCount)
{
    dcgmcm_watch_info_p watchInfo;

    if (!fetchCount)
    {
        PRINT_ERROR("", "fetchCount cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGlobalField(dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *fetchCount = 0;
    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);
    if(watchInfo)
        *fetchCount = watchInfo->fetchCount;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGpuFieldFetchCount(unsigned int gpuId,
                                                     unsigned short dcgmFieldId,
                                                     long long *fetchCount)
{
    dcgmcm_watch_info_p watchInfo;

    if (!fetchCount)
    {
        PRINT_ERROR("", "fetchCount cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGpuField(gpuId, dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *fetchCount = 0;
    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if(watchInfo)
        *fetchCount = watchInfo->fetchCount;

    return DCGM_ST_OK;
}

void DcgmCacheManager::GetRuntimeStats(dcgmcm_runtime_stats_p stats)
{
    if(!stats)
        return;

    m_runStats.lockCount = m_mutex->GetLockCount();
    memcpy(stats, &m_runStats, sizeof(*stats));
}

void DcgmCacheManager::GetValidFieldIds(std::vector<unsigned short> &validFieldIds, bool includeModulePublished)
{
    if(includeModulePublished)
    {
        validFieldIds = m_allValidFieldIds;
        return;
    }

    validFieldIds.clear();

    /* Filter the list for module-published field-IDs */
    for(unsigned int i = 0; i < m_allValidFieldIds.size(); i++)
    {
        if(IsModulePushedFieldId(m_allValidFieldIds[i]))
            continue;
        
        validFieldIds.push_back(m_allValidFieldIds[i]);
    }
}

dcgmReturn_t DcgmCacheManager::GetEntityWatchInfoSnapshot(dcgm_field_entity_group_t entityGroupId,
                                                          dcgm_field_eid_t entityId,
                                                           unsigned int fieldId, dcgmcm_watch_info_p watchInfo)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmcm_watch_info_p foundWatchInfo;

    if(!watchInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    foundWatchInfo = GetEntityWatchInfo(entityGroupId, entityId, fieldId, 0);
    if(foundWatchInfo)
    {
        *watchInfo = *foundWatchInfo; /* Do a deep copy so any sub-objects get properly copied */
    }

    dcgm_mutex_unlock(m_mutex);

    if(!foundWatchInfo)
        return DCGM_ST_NOT_WATCHED;

    return retSt;
}

void DcgmCacheManager::OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection)
{
    int i;
    dcgmcm_watch_info_p watchInfo;
    dcgmReturn_t dcgmReturn;
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);
    dcgm_watch_watcher_info_t watcherInfo;
    watcherInfo.watcher = dcgmWatcher;

    /* Since most users of DCGM have a single daemon / user, it's easy enough just
       to walk every watch in existence and see if the connectionId in question has
       any watches. If we ever have a lot of different remote clients at once, we can
       reevaluate doing this and possibly track watches for each user */

    dcgm_mutex_lock(m_mutex);

    for(void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter; 
        hashIter = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        /* RemoveWatcher will log any failures */
        RemoveWatcher(watchInfo, &watcherInfo, 1);
    }

    dcgm_mutex_unlock(m_mutex);
}

void DcgmCacheManager::WatchVgpuFields(lwmlVgpuInstance_t vgpuId)
{
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeCacheManager);

    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_VM_ID,               3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_VM_NAME,             3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_TYPE,                3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_UUID,                3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_PCI_ID,              30000000,   30.0,   1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_DRIVER_VERSION,      30000000,   30.0,   1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_MEMORY_USAGE,        60000000,   3600.0, 60, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_LICENSE_STATUS,      1000000,    600.0,  600, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT,    900000000,  900.0,  1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_ENC_STATS,           1000000,    600.0,  600, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO,   1000000,    600.0,  600, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FBC_STATS,           1000000,    600.0,  600, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO,   1000000,    600.0,  600, dcgmWatcher, false);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UnwatchVgpuFields(lwmlVgpuInstance_t vgpuId)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgm_field_meta_p fieldMeta = 0;

    /* Remove the VGPU entity and its cached data */
    ClearEntity(DCGM_FE_VGPU, vgpuId, 1);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ColwertVectorToBitmask(std::vector<unsigned int> &gpuIds, uint64_t &outputGpus,
                                              uint32_t numGpus)
{
    outputGpus = 0;

    for (size_t i = 0; i < gpuIds.size() && i < numGpus; i++)
    {
        outputGpus |= 0x1 << gpuIds[i];
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateCpuAffinity(dcgmAffinity_t &affinity)
{
    dcgmcm_sample_t sample = { 0 };
    dcgmReturn_t ret = GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_GPU_TOPOLOGY_AFFINITY, &sample, 0);

    if (ret != DCGM_ST_OK)
    {
        // The information isn't saved
        ret = PopulateTopologyAffinity(affinity);
    }
    else
    {
        dcgmAffinity_t *tmp = (dcgmAffinity_t *)sample.val.blob;
        memcpy(&affinity, tmp, sizeof(affinity));
        free(tmp);
    }
    
    return ret;
}

/*****************************************************************************/
bool DcgmCacheManager::AffinityBitmasksMatch(dcgmAffinity_t &affinity, unsigned int index1, unsigned int index2)
{
    bool match = true;

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        if (affinity.affinityMasks[index1].bitmask[i] != affinity.affinityMasks[index2].bitmask[i])
        {
            match = false;
            break;
        }
    }

    return match;
}

/*****************************************************************************/
void DcgmCacheManager::CreateGroupsFromCpuAffinities(dcgmAffinity_t &affinity,
                                                     std::vector< std::vector<unsigned int> > &affinityGroups,
                                                     std::vector<unsigned int> &gpuIds)
{
    std::set<unsigned int> matchedGpuIds;
    for (unsigned int i = 0 ; i < affinity.numGpus; i++)
    {
        unsigned int gpuId = affinity.affinityMasks[i].dcgmGpuId;
        
        if (matchedGpuIds.find(gpuId) != matchedGpuIds.end())
            continue;

        matchedGpuIds.insert(gpuId);

        // Skip any GPUs not in the input set
        if (std::find(gpuIds.begin(), gpuIds.end(), gpuId) == gpuIds.end())
            continue;

        // Add this gpu as the first in its group and save the index
        std::vector<unsigned int> group;
        group.push_back(gpuId);

        for (unsigned int j = i + 1; j < affinity.numGpus; j++)
        {
            // Skip any GPUs not in the input set
            if (std::find(gpuIds.begin(), gpuIds.end(), affinity.affinityMasks[j].dcgmGpuId) == gpuIds.end())
                continue;

            if (AffinityBitmasksMatch(affinity, i, j) == true)
            {
                unsigned int toAdd = affinity.affinityMasks[j].dcgmGpuId;
                group.push_back(toAdd);
                matchedGpuIds.insert(toAdd);
            }
        }

        affinityGroups.push_back(group);
    }
}

/*****************************************************************************/
void DcgmCacheManager::PopulatePotentialCpuMatches(std::vector<std::vector<unsigned int> > &affinityGroups,
                                                   std::vector<size_t> &potentialCpuMatches, uint32_t numGpus)
{
    for (size_t i = 0; i < affinityGroups.size(); i++)
         
    {
        if (affinityGroups[i].size() >= numGpus)
        {
            potentialCpuMatches.push_back(i);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CombineAffinityGroups(std::vector<std::vector<unsigned int> > &affinityGroups,
                                             std::vector<unsigned int> &combinedGpuList, int remaining)
{
    std::set<unsigned int> alreadyAddedGroups;
    dcgmReturn_t ret = DCGM_ST_OK;

    while (remaining > 0)
    {
        size_t combinedSize = combinedGpuList.size();
        unsigned int largestGroupSize = 0;
        size_t largestGroup = 0;

        for (size_t i = 0; i < affinityGroups.size(); i++)
        {
            // Don't add any group twice
            if (alreadyAddedGroups.find(i) != alreadyAddedGroups.end())
                continue;

            if (affinityGroups[i].size() > largestGroupSize)
            {
                largestGroupSize = affinityGroups[i].size();
                largestGroup = i;

                if (static_cast<int>(largestGroupSize) >= remaining)
                    break;
            }
        }

       alreadyAddedGroups.insert(largestGroup);
       
       // Add the gpus to the combined vector
       for (unsigned int i = 0; remaining > 0 && i < largestGroupSize; i++)
       {
          combinedGpuList.push_back(affinityGroups[largestGroup][i]); 
          remaining--;
       }

       if (combinedGpuList.size() == combinedSize)
       {
           // We didn't add any GPUs, break out of the loop
           ret = DCGM_ST_INSUFFICIENT_SIZE;
           break;
       }
    }

    return ret;
}

/*****************************************************************************/
dcgmTopology_t *DcgmCacheManager::GetLwLinkTopologyInformation()
{
    unsigned int    topologySize = 0;
    dcgmTopology_t *topPtr = NULL;
    dcgmcm_sample_t sample;
    
    dcgmReturn_t ret = GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_GPU_TOPOLOGY_LWLINK, &sample, 0);

    if (ret != DCGM_ST_OK)
    {
        PopulateTopologyLwLink(&topPtr, topologySize);
    }
    else
    {
        topPtr = (dcgmTopology_t *)sample.val.blob;
    }

    return topPtr;
}

/*****************************************************************************/
/*
 * Translate each bitmap into the number of LwLinks that connect the two GPUs
 */
unsigned int DcgmCacheManager::LwLinkScore(dcgmGpuTopologyLevel_t path)
{
    unsigned long temp = static_cast<unsigned long>(path);
    
    // This code relies on DCGM_TOPOLOGY_LWLINK1 equaling 0x100, so 
    // make the code fail so this gets updated if it ever changes
    temp = temp / 256;
    DCGM_CASSERT(DCGM_TOPOLOGY_LWLINK1 == 0x100, 1);
    unsigned int score = 0;

    for (; temp > 0; score++)
        temp = temp / 2;

    return score;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::SetIOConnectionLevels(std::vector<unsigned int> &affinityGroup,
                                                     dcgmTopology_t *topPtr,
                                                     std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > &connectionLevel)

{
    unsigned int highestScore = 0;
    for (unsigned int elementIndex = 0; elementIndex < topPtr->numElements; elementIndex++)
    {
        unsigned int gpuA = topPtr->element[elementIndex].dcgmGpuA;
        unsigned int gpuB = topPtr->element[elementIndex].dcgmGpuB;

        // Ignore the connection if both GPUs aren't in the list
        if ((std::find(affinityGroup.begin(), affinityGroup.end(), gpuA) != affinityGroup.end()) &&
            (std::find(affinityGroup.begin(), affinityGroup.end(), gpuB) != affinityGroup.end()))
        {
            unsigned int score = LwLinkScore(DCGM_TOPOLOGY_PATH_LWLINK(topPtr->element[elementIndex].path));
            DcgmGpuConnectionPair cp(gpuA, gpuB);

            if (connectionLevel.find(score) == connectionLevel.end())
            {
                std::vector<DcgmGpuConnectionPair> temp;
                temp.push_back(cp);
                connectionLevel[score] = temp;

                if (score > highestScore)
                    highestScore = score;
            }
            else
                connectionLevel[score].push_back(cp);
        }
    }

    return highestScore;
}

bool DcgmCacheManager::HasStrongConnection(std::vector<DcgmGpuConnectionPair> &connections, uint32_t numGpus,
                                           uint64_t &outputGpus)
{
    bool strong = false;
//    std::set<size_t> alreadyConsidered;

    // At maximum, connections can have a strong connection between it's size + 1 gpus.
    if (connections.size() + 1 >= numGpus)
    {
        for (size_t outer = 0; outer < connections.size(); outer++)
        {
            std::vector<DcgmGpuConnectionPair> list;
            list.push_back(connections[outer]);
            // There are two gpus in the first connection
            unsigned int strongGpus = 2;

            for (size_t inner = 0; inner < connections.size(); inner++)
            {
                if (strongGpus >= numGpus)
                    break;

                if (outer == inner)
                    continue;

                for (size_t i = 0; i < list.size(); i++)
                {
                    if (list[i].CanConnect(connections[inner]))
                    {
                        list.push_back(connections[inner]);
                        // If it can connect, then we're adding one more gpu to the group
                        strongGpus++;
                        break;
                    }
                }
            }

            if (strongGpus >= numGpus)
            {
                strong = true;
                for (size_t i = 0; i < list.size(); i++)
                {
                    // Checking for duplicates takes more time than setting a bit again
                    outputGpus |= 0x1 << list[i].gpu1;
                    outputGpus |= 0x1 << list[i].gpu2;
                }
                break;
            }
        }
    }

    return strong;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::RecordBestPath(std::vector<unsigned int> &bestPath,
                                              std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > &connectionLevel,
                                              uint32_t numGpus, unsigned int highestLevel)
{
    unsigned int levelIndex = highestLevel;
    unsigned int score = 0;

    for (; bestPath.size() < numGpus && levelIndex > 0; levelIndex--)
    {
        // Ignore a level if not found
        if (connectionLevel.find(levelIndex) == connectionLevel.end())
            continue;

        std::vector<DcgmGpuConnectionPair> &level = connectionLevel[levelIndex];

        for (size_t i = 0; i < level.size(); i++)
        {
            DcgmGpuConnectionPair &cp = level[i];
            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu1) == bestPath.end())
            {
                bestPath.push_back(cp.gpu1);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;

            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu2) == bestPath.end())
            {
                bestPath.push_back(cp.gpu2);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;
        }
    }

    return score;
}

/*****************************************************************************/
void DcgmCacheManager::MatchByIO(std::vector<std::vector<unsigned int> > &affinityGroups,
                                 dcgmTopology_t *topPtr, std::vector<size_t> &potentialCpuMatches,
                                 uint32_t numGpus, uint64_t &outputGpus)
{
    float           scores[DCGM_MAX_NUM_DEVICES] = { 0 };
    std::vector<unsigned int> bestList[DCGM_MAX_NUM_DEVICES];

    // Clear the output
    outputGpus = 0;

    if (topPtr == NULL)
        return;

    for (size_t matchIndex = 0; matchIndex < potentialCpuMatches.size(); matchIndex++)
    {
        unsigned int highestScore;
        std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > connectionLevel;
        highestScore = SetIOConnectionLevels(affinityGroups[potentialCpuMatches[matchIndex]], topPtr, connectionLevel);

        scores[matchIndex] = RecordBestPath(bestList[matchIndex], connectionLevel, numGpus, highestScore);
    }

    // Choose the level with the highest score and mark it's best path
    int bestScoreIndex = 0;
    for (int i = 1; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (scores[i] > scores[bestScoreIndex])
            bestScoreIndex = i;
    }

    ColwertVectorToBitmask(bestList[bestScoreIndex], outputGpus, numGpus);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SelectGpusByTopology(std::vector<unsigned int> &gpuIds, uint32_t numGpus,
                                                    uint64_t &outputGpus)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    // First, group them by cpu affinity
    dcgmAffinity_t affinity = { 0 };
    std::vector< std::vector<unsigned int> > affinityGroups;
    std::vector<size_t> potentialCpuMatches;

    if (gpuIds.size() <= numGpus)
    {
        // We don't have enough healthy gpus to be picky, just set the bitmap
        ColwertVectorToBitmask(gpuIds, outputGpus, numGpus);
        
        // Set an error if there aren't enough GPUs to fulfill the request
        if (gpuIds.size() < numGpus)
            ret = DCGM_ST_INSUFFICIENT_SIZE;
    }
    else
    {
        ret = PopulateCpuAffinity(affinity);

        if (ret != DCGM_ST_OK)
        {
            return DCGM_ST_GENERIC_ERROR;
        }
            
        CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);

        PopulatePotentialCpuMatches(affinityGroups, potentialCpuMatches, numGpus);

        if ((potentialCpuMatches.size() == 1) && 
            (affinityGroups[potentialCpuMatches[0]].size() == numGpus))
        {
            // CPUs have already narrowed it down to one match, so go with that.
            ColwertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
        }
        else if (potentialCpuMatches.size() == 0)
        {
            // Not enough GPUs with the same CPUset
            std::vector<unsigned int> combined;
            ret = CombineAffinityGroups(affinityGroups, combined, numGpus);
            if (ret == DCGM_ST_OK)
                ColwertVectorToBitmask(combined, outputGpus, numGpus);
        }
        else
        {
            // Find best interconnect within or among the matches.
            dcgmTopology_t *topPtr = GetLwLinkTopologyInformation();
            if (topPtr != NULL)
            {
                MatchByIO(affinityGroups, topPtr, potentialCpuMatches, numGpus, outputGpus);
                free(topPtr);
            }
            else
            {
                // Couldn't get the LwLink information, just pick the first potential match
                PRINT_DEBUG("", "Unable to get LwLink topology, selecting solely based on cpu affinity");
                ColwertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
            }
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateLwLinkLinkStatus(dcgmLwLinkStatus_v1 &lwLinkStatus)
{
    int j;

    lwLinkStatus.version = dcgmLwLinkStatus_version1;

    for(unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmcmGpuStatusDetached)
            continue;

        /* Make sure the GPU LwLink states are up to date before we return them to users */
        UpdateLwLinkLinkState(m_gpus[i].gpuId);

        lwLinkStatus.gpus[i].entityId = m_gpus[i].gpuId;
        for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_GPU_LEGACY1; j++)
        {
            lwLinkStatus.gpus[i].linkState[j] = m_gpus[i].lwLinkLinkState[j];
        }
    }
    lwLinkStatus.numGpus = m_numGpus;

    for(int i = 0; i < m_numLwSwitches; i++)
    {
        lwLinkStatus.lwSwitches[i].entityId = m_lwSwitches[i].physicalId;
        for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; j++)
        {
            lwLinkStatus.lwSwitches[i].linkState[j] = m_lwSwitches[i].lwLinkLinkState[j];
        }
    }
    lwLinkStatus.numLwSwitches = m_numLwSwitches;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::PopulateLwLinkLinkStatus(dcgmLwLinkStatus_v2 &lwLinkStatus)
{
    int j;

    lwLinkStatus.version = dcgmLwLinkStatus_version2;

    for(unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmcmGpuStatusDetached)
            continue;

        /* Make sure the GPU LwLink states are up to date before we return them to users */
        UpdateLwLinkLinkState(m_gpus[i].gpuId);

        lwLinkStatus.gpus[i].entityId = m_gpus[i].gpuId;
        for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_GPU; j++)
        {
            lwLinkStatus.gpus[i].linkState[j] = m_gpus[i].lwLinkLinkState[j];
        }
    }
    lwLinkStatus.numGpus = m_numGpus;

    for(int i = 0; i < m_numLwSwitches; i++)
    {
        lwLinkStatus.lwSwitches[i].entityId = m_lwSwitches[i].physicalId;
        for(j = 0; j < DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH; j++)
        {
            lwLinkStatus.lwSwitches[i].linkState[j] = m_lwSwitches[i].lwLinkLinkState[j];
        }
    }
    lwLinkStatus.numLwSwitches = m_numLwSwitches;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetEntityLwLinkLinkStatus(dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         dcgmLwLinkLinkState_t *linkStates)
{
    int i;

    if((entityGroupId != DCGM_FE_GPU && entityGroupId != DCGM_FE_SWITCH) || !linkStates)
    {
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }
    

    if(entityGroupId == DCGM_FE_GPU)
    {
        if (entityId < 0 || entityId >= m_numGpus)
        {
            PRINT_ERROR("%u", "Invalid gpuId %u", entityId);
            return DCGM_ST_BADPARAM;
        }

        /* Make sure the GPU LwLink states are up to date before we return them to users */
        UpdateLwLinkLinkState(entityId);

        memcpy(linkStates, m_gpus[entityId].lwLinkLinkState, sizeof(m_gpus[entityId].lwLinkLinkState));
    }
    else /* LwSwitch. Already validated at top of function */
    {
        dcgmcm_lwswitch_info_t *lwSwitch = NULL;
        for(i = 0; i < m_numLwSwitches; i++)
        {
            if(m_lwSwitches[i].physicalId == entityId)
            {
                lwSwitch = &m_lwSwitches[i];
                break;
            }
        }
        if(!lwSwitch)
        {
            PRINT_ERROR("%u", "Invalid LwSwitch entityId %u", entityId);
            return DCGM_ST_BADPARAM;
        }

        memcpy(linkStates, lwSwitch->lwLinkLinkState, sizeof(lwSwitch->lwLinkLinkState));
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
/*****************************************************************************/
/* DcgmCacheManagerEventThread methods */
/*****************************************************************************/
/*****************************************************************************/
DcgmCacheManagerEventThread::DcgmCacheManagerEventThread(DcgmCacheManager *cacheManager) : LwcmThread(false)
{
    m_cacheManager = cacheManager;
}

/*****************************************************************************/
DcgmCacheManagerEventThread::~DcgmCacheManagerEventThread(void)
{
}

/*****************************************************************************/
void DcgmCacheManagerEventThread::run(void)
{
    PRINT_INFO("", "DcgmCacheManagerEventThread started");

    while(!ShouldStop())
    {
        m_cacheManager->EventThreadMain(this);
    }

    PRINT_INFO("", "DcgmCacheManagerEventThread ended");
}

/*****************************************************************************/
