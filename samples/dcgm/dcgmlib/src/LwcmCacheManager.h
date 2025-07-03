#ifndef DCGMCACHEMANAGER_H
#define DCGMCACHEMANAGER_H

#include "DcgmFvBuffer.h"
#include "DcgmMutex.h"
#include "DcgmWatcher.h"
#include "LwcmServerConnection.h"
#include "LwcmSettings.h"
#include "LwcmThread.h"
#include "lwos.h"
#include "dcgm_fields.h"
#include "dcgm_fields_internal.h"
#include "dcgm_structs.h"
#include "hashtable.h"
#include "lwml.h"
#include "lwml_grid.h"
#include "lwml_internal.h"
#include "timelib.h"
#include "timeseries.h"
#include <bitset>
#include <map>
#include <set>
#include <string>
#include <tr1/unordered_map>

/*****************************************************************************/
typedef enum
{
    DcgmcmGpuStatusUnknown = 0, /* GPU has not been referenced yet */
    DcgmcmGpuStatusOk,          /* Gpu is known and OK */
    DcgmcmGpuStatusUnsupported, /* Gpu is unsupported by DCGM */
    DcgmcmGpuStatusInaccessible,/* Gpu is inaccessible, usually due to cgroups */
    DcgmcmGpuStatusGpuLost,     /* Gpu has been lost. Usually set from LWML
                                   returning LWML_ERROR_GPU_IS_LOST */
    DcgmcmGpuStatusFakeGpu,     /* Gpu is a fake, injection-only GPU for testing */
    DcgmcmGpuStatusDisabled,    /* Don't collect values from this GPU */
    DcgmcmGpuStatusDetached     /* Gpu is detached, not good for any uses */
} DcgmcmGpuStatus_t;

/*****************************************************************************/
/* Summary information types */
typedef enum
{
    DcgmcmSummaryTypeMinimum = 0,   /* Minimum value */
    DcgmcmSummaryTypeMaximum,       /* Maximum value */
    DcgmcmSummaryTypeAverage,       /* Simple Average (sum / Nelem) */
    DcgmcmSummaryTypeSum,           /* Sum of the values */
    DcgmcmSummaryTypeCount,         /* Count of the number of values */
    DcgmcmSummaryTypeIntegral,      /* Integral of the values (area under values) */
    DcgmcmSummaryTypeDifference,    /* Difference between the first and last
                                       non-blank value (last - first) */

    DcgmcmSummaryTypeSize           /* Always last entry */
} DcgmcmSummaryType_t;

/*****************************************************************************/
typedef struct dcgmcm_sample_t /* This is made to look similar to timeseries_value_t
                                  since that is the underlying data anyway */
{
    timelib64_t timestamp; /* Value timestamp in usec since 1970 */

    union
    {
        double d;      /* Double value */
        long long i64; /* Int 64 value */
        char *str;     /* Null-terminated pointer to allocated memory for strings
                          must be freed if populated */
        void *blob;    /* Allocated memory for blob types. Must be freed if
                          populated */
    } val;

    union
    {
        double d;      /* Double value */
        long long i64; /* Int 64 value */
        long long ptrSize; /* Size of allocated memory if str or blob is populated.
                              Includes null terminator for str type */
    } val2; /* Possible second value. This should only be used for string and
               blob types to contain the size of the memory str/blob points at */

} dcgmcm_sample_t, *dcgmcm_sample_p;

/*****************************************************************************/
/* Details for a single watcher of a field. Each fieldId has a vector of these */
typedef struct dcgm_watch_watcher_info_t
{
    DcgmWatcher watcher;                /* Who owns this watch? */
    timelib64_t monitorFrequencyUsec;   /* How often this field should be sampled */
    timelib64_t maxAgeUsec;             /* Maximum time to cache samples of this
                                           field. If 0, the class default is used */
    int         isSubscribed;           /* Does this watcher want live updates
                                           when this field value updates? */
} dcgm_watch_watcher_info_t, *dcgm_watch_watcher_info_p;

/*****************************************************************************/
/* Unique key for a given fieldEntityGroup + entityId + fieldId combination */
typedef struct dcgmcm_entity_key_t
{
    dcgm_field_eid_t entityId;          /* Entity ID of this watch */
    unsigned short fieldId;             /* Field ID of this watch */
    unsigned short entityGroupId;       /* DCGM_FE_? #define of the entity group this 
                                           belongs to */
} dcgmcm_entity_key_t; /* 8 bytes */

/*****************************************************************************/
/*
 * Struct to hold a single watch and its field values
 * 
 * IMPORTANT: If you add fields to this struct, update AllocWatchInfo() to 
 *            initialize those fields properly. Because we are allocating the
 *            struct with new() and the struct has classes (watchers), we can't
 *            simply memset the struct to zero
 */
typedef struct dcgmcm_watch_info_t
{
    dcgmcm_entity_key_t watchKey;       /* Key information for this watch */
    short isWatched;                    /* Is this field being watched. 1=yes. 0=no */
    short hasSubscribedWatchers;        /* Does this field have any watchers that
                                           have subscribed for notifications?. This
                                           should be the logical OR of 
                                           watchers[0-n].isSubscribed */
    lwmlReturn_t lastStatus;            /* Last status returned from querying this
                                           value. See LWML_? values in lwml.h */
    timelib64_t lastQueriedUsec;        /* Last time we updated this value. Used for
                                           determining if we should request an update
                                           of this field or not */
    timelib64_t monitorFrequencyUsec;   /* How often this field should be sampled */
    timelib64_t maxAgeUsec;             /* Maximum time to cache samples of this
                                           field. If 0, the class default is used */
    timelib64_t execTimeUsec;           /* Cumulative time spent updating this
                                           field since the cache manager started */
    long long fetchCount;               /* Number of times that this field has been
                                           fetched from the driver */
    timeseries_p timeSeries;            /* Time-series of values for this watch */
    std::vector<dcgm_watch_watcher_info_t>watchers; /* Info for each watcher of this
                                                       field. monitorFrequencyUsec and
                                                       maxAgeUsec come from this array */
} dcgmcm_watch_info_t, *dcgmcm_watch_info_p;

/*****************************************************************************/
typedef struct dcgmcm_vgpu_info_t
{
    lwmlVgpuInstance_t vgpuId;          /* vGPU Instance ID */
    bool found;                         /* Flag to denote that the vGPU instance is
                                           still active */
    struct dcgmcm_vgpu_info_t *next;    /* Pointer to the next node in the list */
} dcgmcm_vgpu_info_t, *dcgmcm_vgpu_info_p;

/*****************************************************************************/
typedef struct dcgmcm_gpu_info_t
{
    unsigned int gpuId;             /* ID of this GPU */
    DcgmcmGpuStatus_t status;       /* Status of this GPU */

    /* Cached LWML fields */
    unsigned int lwmlIndex;         /* LWML index */
    lwmlDevice_t lwmlDevice;        /* LWML handle. This is lwrrently a static pointer.
                                       ok to cache*/
    char uuid[128];                 /* UUID */
    dcgmGpuBrandType_t brand;       /* Brand */
    lwmlPciInfo_t pciInfo;          /* PCI Info */
    lwmlChipArchitecture_t arch;    /* chip architecture */

    /* vGPU Instance metadata */
    dcgmcm_vgpu_info_p vgpuList;    /* Head pointer of the vGPU linked list for this GPU */

    /* LwLink per-lane status */
    dcgmLwLinkLinkState_t lwLinkLinkState[DCGM_LWLINK_MAX_LINKS_PER_GPU];

    bool lwLinkCountersAreFlits;    /* Are our LwLink counters returned in flits (true) or
                                       bytes (false). This affects the multiplier we need
                                       to use on the returned values */

} dcgmcm_gpu_info_t, *dcgmcm_gpu_info_p;

/*****************************************************************************/
/* Version of dcgmcm_gpu_info_t that can be returned to other modules via
   DcgmCacheManager::GetAllGpuInfo */
typedef struct dcgmcm_gpu_info_cached_t
{
    unsigned int gpuId;             /* ID of this GPU */
    DcgmcmGpuStatus_t status;       /* Status of this GPU */
    dcgmGpuBrandType_t brand;       /* Brand */
    unsigned int lwmlIndex;         /* LWML index */
    char uuid[128];                 /* UUID */
    lwmlPciInfo_t pciInfo;          /* PCI Info */
    lwmlChipArchitecture_t arch;    /* chip architecture */
    /* if you add fields to this, make sure they're handled in 
       DcgmCacheManager::GetAllGpuInfo */
} dcgmcm_gpu_info_cached_t, *dcgmcm_gpu_info_cached_p;

/*****************************************************************************/
typedef struct dcgmcm_lwswitch_info_t
{
    unsigned int physicalId;   /* Physical hardware ID of the LwSwitch 
                                  (expected values are 0x10 (10000)  to 0x15 (10101) 
                                  for board1 LWSwitches and 0x18(11000) to 0x1D (11101) 
                                  for board2 LWSwitches) */
    DcgmcmGpuStatus_t status;  /* Status of this LwSwitch */
    dcgmLwLinkLinkState_t lwLinkLinkState[DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH]; /* LwLink link state 
                                                                                  for each link */
} dcgmcm_lwswitch_info_t, *dcgmcm_lwswitch_info_p;

/*****************************************************************************/
/* Runtime stats for the cache manager */
typedef struct dcgmcm_runtime_stats_t
{
    long long numSleepsSkipped; /* Number of times the cache manager update thread skipped sleeping due
                                   to the update loop taking too long */
    long long numSleepsDone;    /* Number of times the cache manager thread slept for any duration */
    long long sleepTimeUsec;    /* Amount of time the cache manager update thread was sleeping in usec */
    long long awakeTimeUsec;    /* Amount of time the cache manager update thread was awake in usec */

    long long updateCycleStarted;  /* Counter of how many update cycles have been started */
    long long updateCycleFinished; /* Counter of how many update cycles have been finished */
    long long shouldFinishCycle;   /* Lockstep mode only. How many cycles should we be
                                      finished with? If m_updateCycleFinished < m_shouldFinishCycle,
                                      we start another cycle loop. This is technically the
                                      predicate for m_condition */
    long long lockCount;           /* Number of times that the cache manager mutex has been locked. This is
                                      periodically snapshotted from the cache manager update threads */
} dcgmcm_runtime_stats_t, *dcgmcm_runtime_stats_p;

/*****************************************************************************/
/* Cache manager update thread context structure. This exists to prevent stack
   overflow in the update thread by storing large objects on the heap */
typedef struct dcgmcm_update_thread_t
{
    /* Information about the entity lwrrently being worked on */
    dcgmcm_entity_key_t entityKey; /* Key information for the current entity */
    dcgmcm_watch_info_p watchInfo; /* Optional cached pointer to the watch object for this entity.
                                      If provided, any updates will be cached here */
    
    DcgmFvBuffer *fvBuffer;        /* If != NULL, this any Append* calls will result in
                                      values being appended to this structure */
    unsigned int affectedSubscribers; /* This is a bitmask of DcgmWatcherType_t bits of which watchers have
                                         suscribed-for updates in fvBuffer. Using a bitmask here to avoid
                                         zeroing a large array of ints every update loop, also not using
                                         a std::set or list of DcgmWatcherType_t's for performance.
                                         Finally, we're not tracking clientId yet. If we ever extend this
                                         functionality to clients, we will have to track client ID as well. */

    int numFieldValues[DCGM_MAX_NUM_DEVICES]; /* Number of field values per device
                                                 saved in fieldValueFields and fieldValueWatchInfo*/
    dcgm_field_meta_p fieldValueFields[DCGM_MAX_NUM_DEVICES][LWML_FI_MAX]; /* Fields to update with field-value APIs rather than CacheLatest*Value() */
    dcgmcm_watch_info_p fieldValueWatchInfo[DCGM_MAX_NUM_DEVICES][LWML_FI_MAX]; /* Watch info for field values */
} dcgmcm_update_thread_t, *dcgmcm_update_thread_p;

/*****************************************************************************/
/* Callback function
 * Return True if the entry should be used for summary callwlation
 * Return False to filter out entry from summary callwlation
 */
 typedef bool (*pfUseEntryForSummary)(timeseries_entry_p entry, void *userData);

/******************************************************************************
 *
 * This is a callback to provide DcgmCacheManager to be called when a subscribed
 * field value is updated by the cache manager.
 * 
 * fvBuffer        IN: Contains buffered field values that have updated that subscribers
 *                     have said they cared about. It's the the callee's job to walk the
 *                     FVs and determine if they need the updates or not.
 * watcherTypes    IN: Which watchers care about the updates in fvBuffer
 * numWatcherTypes IN: How many entries in watcherTypes are valid
 * 
 * userData IN: A user-supplied pointer that was passed to 
 * DcgmCacheManager::SubscribeForFvUpdates
 */
typedef void (*dcgmOnSubscribedFvUpdate_f)(DcgmFvBuffer *fvBuffer, DcgmWatcherType_t *watcherTypes, 
                                           int numWatcherTypes, void *userData);

/* Array entry to track each callback that has been registered */
typedef struct 
{
    dcgmOnSubscribedFvUpdate_f callback;
    void *userData;
} dcgmSubscribedFvUpdateCBEntry_t;

/*****************************************************************************/
class DcgmCacheManager; /* Forward declare the cache manager so it can be 
                           used by DcgmCacheManagerEventThread */

/*****************************************************************************/
/* This class handles reading LWML events like XIDs */
class DcgmCacheManagerEventThread : public LwcmThread
{
private:
    DcgmCacheManager *m_cacheManager; /* Pointer to the cache manager instance
                                         we belong to */

public:
    DcgmCacheManagerEventThread(DcgmCacheManager *cacheManager);
    ~DcgmCacheManagerEventThread(void);

    /*************************************************************************/
    /*
     * Inherited virtual method from DcgmThread. This is the method that
     * actually does the work
     *
     */
    void run(void);
};

class DcgmGpuConnectionPair
{
public:
    unsigned int gpu1;
    unsigned int gpu2;

    DcgmGpuConnectionPair(unsigned int g1, unsigned int g2) : gpu1(g1), gpu2(g2)
    {
    }

    bool operator ==(const DcgmGpuConnectionPair &other) const
    {
        return ((other.gpu1 == this->gpu1 && other.gpu2 == this->gpu2) ||
                (other.gpu2 == this->gpu1 && other.gpu1 == this->gpu2));
    }

    bool CanConnect(const DcgmGpuConnectionPair &other) const
    {
        return ((this->gpu1 == other.gpu1) ||
                (this->gpu1 == other.gpu2) ||
                (this->gpu2 == other.gpu1) ||
                (this->gpu2 == other.gpu2));
    }

    bool operator <(const DcgmGpuConnectionPair &other) const
    {
        return this->gpu1 < other.gpu1;
    }
};

/*****************************************************************************/
/* Cache manager main class */
class DcgmCacheManager : public LwcmThread
{
public:
    /* Constructor/destructor */
    DcgmCacheManager();
    ~DcgmCacheManager();

    /*************************************************************************/
    /*
     * Initialize the cache manager, returning errors if they occur on
     * initialization
     *
     * pollInLockStep IN: Whether to poll when told to (1) or at the frequency
     *                    of the most frequent field being watched (0)
     *                    If 1, you will need to force stat collection to occur
     *                    by calling UpdateAllFields()
     * maxSampleAge   IN: Maximum time in seconds to keep a sample within this
     *                    module.
     *
     * Returns 0 on Success
     *        <0 on error
     */
    dcgmReturn_t Init(int pollInLockStep, double maxSampleAge);

    /*************************************************************************/
    /*
     * Shutdown and clean up this object, including stopping the monitoring thread
     *
     * saveToFilename IN: If not "", is the filename to serialize our stat cache to
     *
     * Returns 0 on Success
     *        <0 on error
     *
     */
    dcgmReturn_t Shutdown(std::string saveToFilename);

    /*************************************************************************/
    /*
     * Save the cache to a file in JSON format
     *
     * Returns 0 on success.
     *        <0 on error.
     *
     */
    dcgmReturn_t SaveCache(std::string filename);

    /*************************************************************************/
    /*
     * Load the cache from a JSON file, replacing the contents of the cache with the
     * stats parsed from filename
     *
     * Returns 0 on success.
     *        <0 on error.
     *
     */
    dcgmReturn_t LoadCache(std::string filename);

    /*************************************************************************/
    /* Empty the contents of the stats cache
     *
     * Returns 0 on success.
     *        <0 on error.
     */
    dcgmReturn_t EmptyCache();

    /*************************************************************************/
    /*
     * Start the the field sampling thread. Call AddFieldWatch for each field
     * you want to be recorded in the cache after this call to enable
     * sampling.
     *
     * Returns: 0 on success
     *         <0 on error
     *
     */
    dcgmReturn_t Start();

    /*************************************************************************/
    /*
     * Subscribe for cache manager updates. updateCB will be called whelwer
     * field values that have been subscribed to update. In order to receive
     * notifications via this callback, you should pass subscribeForUpdates=true
     * to AddFieldWatch().
     * 
     * updateCB  IN: Callback to call
     * userData  IN: Pointer to pass to updateCB when it is called.
     * 
     */
    dcgmReturn_t SubscribeForFvUpdates(dcgmOnSubscribedFvUpdate_f updateCB, void *userData);

    /*************************************************************************/
    /*
     * Get info about all of the GPUs the cache manager knows about
     * 
     * Returns DCGM_ST_OK on success
     *         Other DCGM_ST_? #define on error
     */
    dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo);

    /*************************************************************************/
    /*
     * Get the number of GPUs seen by DCGM
     *
     * activeOnly: If set to 1, will not count GPUs that are inaccessible for any
     *             reason. Reasons include (but are not limited to):
     *                 - not being whitelisted
     *                 - being blocked by cgroups
     *                 - fallen off bus
     *
     * Returns: Count of GPUs
     */
    int GetGpuCount(int activeOnly);

    /*************************************************************************/
    /*
     * Get the gpuIds of the GPUs seen by DCGM
     *
     * activeOnly: If set to 1, will not count GPUs that are inaccessible for any
     *             reason. Reasons include (but are not limited to):
     *                 - not being whitelisted
     *                 - being blocked by cgroups
     *                 - fallen off bus
     * gpuIds: Vector of unsigned ints to fill with GPU IDs.
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error
     */
    dcgmReturn_t GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds);

    /*************************************************************************/
    /*
     * Get the entities seen by DCGM of a given entityGroupId
     *
     * activeOnly     IN: If set to 1, will not count GPUs that are inaccessible for any
     *                    reason. Reasons include (but are not limited to):
     *                    - not being whitelisted
     *                    - being blocked by cgroups
     *                    - fallen off bus
     * entityGroupId IN: Which entity group to fetch the entities of
     * entities     OUT: Entity IDs
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error
     */
    
    dcgmReturn_t GetAllEntitiesOfEntityGroup(int activeOnly, 
                                             dcgm_field_entity_group_t entityGroupId, 
                                             std::vector<dcgmGroupEntityPair_t> &entities);

    /*************************************************************************/
    /*
     * Get the status of an entity
     *
     * entityGroupId is the group the entity belongs to
     * entityId is the ID of the entity
     *
     * Returns a GPU status enum
     *
     */
    DcgmcmGpuStatus_t GetEntityStatus(dcgm_field_entity_group_t entityGroupId, 
                                      dcgm_field_eid_t entityId);

    /*************************************************************************/
    /*
     * Get the status of a GPU
     *
     * gpuId is the ID of the GPU to get the status of
     *
     * Returns a GPU status enum
     *
     */
    DcgmcmGpuStatus_t GetGpuStatus(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Get the brand of a given GPU
     *
     * Returns: DCGM_GPU_BRAND_UNKNOWN if the gpuId is invalid
     *          Otherwise the DCGM_GPU_BRAND_* of the GPU found. 
     */
    dcgmGpuBrandType_t GetGpuBrand(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Get the GPU Chip Architecture information
     *
     * gpuId is the ID of the GPU to get the Architecture of
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error
     *
     */
    dcgmReturn_t GetGpuArch(unsigned int gpuId, lwmlChipArchitecture_t &arch);

    /*************************************************************************/
    /*
     * Performs a GPU Reset
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error.
     */
    dcgmReturn_t GpuReset(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Get the list of GPUs that were blacklisted by the driver
     * 
     * Note: This will only work on r400 or newer drivers. Otherwise, an empty
     *       list will always be returned.
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error.
     */
    dcgmReturn_t GetGpuBlacklist(std::vector<lwmlBlacklistDeviceInfo_t> &blacklist);

    /*************************************************************************/
    /*
     * Inherited virtual method from DcgmThread. This is the method that
     * actually does the work
     *
     */

    void run(void);

    /*************************************************************************/
    /*
     * Pause or resume field sampling for a given GPU or all GPUs
     *
     * If pausing, this function blocks until the pause has been confirmed by
     * the sampling loop so that the sampling loop doesn't get unexpected errors
     * and the caller can safely do things like reset the GPU.
     *
     * gpuId    IN: Which GPU to pause/resume on
     *
     * Returns: 0 on success
     *          DCGM_ST_? #define on error.
     *
     *
     */
    dcgmReturn_t PauseGpu(unsigned int gpuId);
    dcgmReturn_t ResumeGpu(unsigned int gpuId);
    dcgmReturn_t Pause();
    dcgmReturn_t Resume();

    /*************************************************************************/
    /*
     * This notifies the sampling thread that
     * it should do a round of sampling before going to sleep again.
     *
     * waitForUpdate IN: Whether (1) or not (0) the caller should wait for the
     *                   triggered update cycle to finish before returning.
     *
     * Returns 0 on success
     *         DCGM_ST_? #define on error.
     *
     */
    dcgmReturn_t UpdateAllFields(int waitForUpdate);


    /*************************************************************************/
    /*
     * Add watching of a field by this class. Samples will be gathered every
     * monitorFrequencyUsec usec. Samples will only be kept for a maximum of
     * maxSampleAge seconds before being discarded.
     *
     * If a field is already watched the resulting monitor frequency and
     * maxSampleAge will be the maximum of what is lwrrently set and what is
     * passed in on the new watch
     *
     * entityGroupId        IN: Which entity group this watch pertains to
     * entityId             IN: Which entity ID this watch pertains to
     * dcgmFieldId          IN: Which DCGM field to add the watch for
     * monitorFrequencyUsec IN: How often in usec to gather samples for this field.
     *                          0=use default from DcgmField metadata
     * maxSampleAge         IN: How long to keep samples for in seconds.
     *                          0.0 = use default from DcgmField metadata
     * maxKeepSamples       IN: Maximum number of samples to keep.
     *                          0=no limit
     * watcher              IN: Who is watching this field? Used for tracking purposes
     * subscribeForUpdates  IN: Whether watcher wants to receive notification callbacks
     *                          whenever this field value updates
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     *
     */
    dcgmReturn_t AddFieldWatch(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                               unsigned short dcgmFieldId, timelib64_t monitorFrequencyUsec, double maxSampleAge,
                               int maxKeepSamples, DcgmWatcher watcher, bool subscribeForUpdates);

    /*************************************************************************/
    /*
     * Update the caching frquency, maxSampleAge and maxKeepSamples for a given field
     * into watchInfo.
     *
     * monitorFrequencyUsec IN: How often in usec to gather samples for this field.
     *                          0=use default from DcgmField metadata
     * maxSampleAge         IN: How long to keep samples for in seconds.
     *                          0.0 = use default from DcgmField metadata
     * maxKeepSamples       IN: Maximum number of samples to keep.
     *                          0=no limit
     * watcher              IN: Who is watching this field? Used for tracking purposes
     *
     * Returns 0                   on success
     *         DCGM_ST_NOT_WATCHED on field not watched.
     *         DCGM_ST_BADPARAM    on NULL watchInfo.
     *
     */
    dcgmReturn_t UpdateFieldWatch(dcgmcm_watch_info_p watchInfo, timelib64_t updatedMonitorFrequencyUsec,
                                  double maxSampleAge, int maxKeepSamples, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Get the frequency that a field is watched at.
     *
     * gpuId        IN: Which GPU the watch pertains to. This is ignored if the field is not a GPU field.
     * fieldId      IN: Which DCGM field the watch is for
     * freqUsec    OUT: How often in usec to gather samples for this field.
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t GetFieldWatchFreq(unsigned int gpuId, unsigned short fieldId, timelib64_t *freqUsec);

    /*************************************************************************/
    /*
     * Check if a GPU field is being watched.
     *
     * gpuId        IN: Which GPU the watch pertains to.
     * dcgmFieldId  IN: Which DCGM field the watch is for
     * isWatched   OUT: If the field is lwrrently watched
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t IsGpuFieldWatched(unsigned int gpuId, unsigned short dcgmFieldId,
                                   bool *isWatched);

    /*************************************************************************/
    /*
     * Check if the given field is watched on any GPU.
     *
     * dcgmFieldId  IN: Which DCGM field the watch is for
     * isWatched   OUT: If the field is lwrrently watched
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t IsGpuFieldWatchedOnAnyGpu(unsigned short fieldId, bool *isWatched);

    /*************************************************************************/
    /*
     * Check if a global field is being watched.
     *
     * dcgmFieldId  IN: Which DCGM field the watch is for
     * isWatched   OUT: If the field is lwrrently watched
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t IsGlobalFieldWatched(unsigned short dcgmFieldId, bool *isWatched);

    /*************************************************************************/
    /*
     * Check if any of the given fields are watched.
     *
     * dcgmFieldIds  IN: Which DCGM fields to check. NULL means all fields.
     * isWatched    OUT: If the field is lwrrently watched
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    bool AnyGlobalFieldsWatched(std::vector<unsigned short> *fieldIds);

    /*************************************************************************/
    /*
     * Check if any of the given fields are watched on the given GPU.
     *
     * gpuId         IN: the GPU to check for field watches
     * dcgmFieldIds  IN: Which DCGM fields to check. NULL means all fields.
     * isWatched    OUT: If the field is lwrrently watched
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    bool AnyGpuFieldsWatched(unsigned int gpuId, std::vector<unsigned short> *fieldIds);

    /*************************************************************************/
   /*
    * Check if any of the given fields are watched on any GPU.
    *
    * dcgmFieldIds  IN: Which DCGM fields to check. NULL means all fields.
    * isWatched    OUT: If the field is lwrrently watched
    *
    * Returns 0 on success
    *        <0 on error. See DCGM_ST_? #defines
    */
    bool AnyGpuFieldsWatchedAnywhere(std::vector<unsigned short> *fieldIds);

    bool AnyFieldsWatched(std::vector<unsigned short> *fieldIds);

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each GPU field that is lwrrently watched.
     *
     * callbackFn  IN: function to be called. Must have signature
     *                 "dcgmReturn_t fn(unsigned short fieldId, unsigned int gpuId)"
     * fieldIds    IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGpuField(Fn *callbackFn,
                                        std::vector<unsigned short>* fieldIds=NULL,
                                        unsigned int * const onlyOnThisGpuId=NULL);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each GPU field that is lwrrently watched.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId, unsigned int gpuId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGpuField(std::vector<Fn *> callbackFns,
                                        std::vector<unsigned short>* fieldIds=NULL,
                                        unsigned int * const onlyOnThisGpuId=NULL);

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each global field that is lwrrently watched.
     *
     * callbackFn  IN: function to be called. Must have signature
     *                 "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds    IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGlobalField(Fn *callbackFn,
                                           std::vector<unsigned short>* fieldIds=NULL);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each global field that is lwrrently watched.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedGlobalField(std::vector<Fn *> callbackFns,
                                           std::vector<unsigned short>* fieldIds=NULL);

    /*************************************************************************/
    /*
     * Execute function (usually a functor) for each field that is lwrrently watched globally
     * or on at least 1 GPU.
     *
     * callbackFns  IN: function to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedField(Fn * callbackFn,
                                     std::vector<unsigned short>* fieldIds=NULL);

    /*************************************************************************/
    /*
     * Execute functions (usually functors) for each field that is lwrrently watched globally
     * or on at least 1 GPU.
     *
     * callbackFns  IN: functions to be called. Must have signature
     *                  "dcgmReturn_t fn(unsigned short fieldId)"
     * fieldIds     IN: (optional) a subset of fields to iterate over instead of all fields
     *
     * Returns 0 on success
     *        <0 on error for the last error that oclwrred when calling the callbackFn's. See DCGM_ST_? #defines
     */
    template <typename Fn>
    dcgmReturn_t ForEachWatchedField(std::vector<Fn *> callbackFns,
                                     std::vector<unsigned short>* fieldIds=NULL);

    /*************************************************************************/
    /*
     * Stop and remove monitoring of a field for a GPU
     *
     * entityGroupId IN: Which entity group to remove the watch for
     * entityId      IN: Which entity within entityGroupId to remove the watch for
     * dcgmFieldId   IN: Which DCGM field to remove the watch for
     * clearCache    IN: Whether (1) or not (0) to clear cached data after
     *                   we've stopped monitoring data
     * watcher       IN: Who is watching this field? Used for tracking purposes
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     *
     */
    dcgmReturn_t RemoveFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                  int clearCache, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Get the latest instance of a PID's accounting data from the cache manager
     *
     * gpuId     IN: Which GPU to get the value for
     * pid       IN: Process ID to get accounting data for
     * pidInfo  OUT: Process information for pid. Only valid if this function returns 0
     *
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     *
     */
    dcgmReturn_t GetLatestProcessInfo(unsigned int gpuId, unsigned int pid,
                                      dcgmDevicePidAccountingStats_t *pidInfo);

    /*************************************************************************/
    /*
     * Get a list of unique graphics or compute pids
     *
     * entityGroupId   IN: Which entity group this entity belongs to
     * entityId        IN: Which entity to get PIDs for
     * dcgmfieldId     IN: Field ID to get the PID list from. Lwrrently supported fields:
     *                          DCGM_FI_DEV_GRAPHICS_PIDS
     *                          DCGM_FI_DEV_COMPUTE_PIDS
     * excludePid      IN: Optional PID to exclude from the list. 0=no exclusion
     * pids           OUT: PIDs. numPids contains the count
     * numPids         IO: On calling, contains the capacity of pids. The output
     *                     is the number of entries set in pids
     * startTime       IN: Optional starting timestamp. Samples >= this value will
     *                     be returned. 0=Return samples from beginning
     * endTime         IN: Optional ending timestamp. Samples <= this value will
     *                     be returned. 0=Return samples up until end
     *
     * Returns: 0 on success.
     *          DCGM_ST_? enum on error.
     */
    dcgmReturn_t GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                   unsigned int excludePid,
                                   dcgmProcessUtilInfo_t *pidInfo, unsigned int *numPids,
                                   timelib64_t startTime, timelib64_t endTime);

    /*************************************************************************/
    /*
     * Get a list of unique graphics or compute pids
     *
     * entityGroupId   IN: Which entity group this entity belongs to
     * entityId        IN: Which entity to get PIDs for
     * dcgmfieldId     IN: Field ID to get the PID list from. Lwrrently supported fields:
     *                          DCGM_FI_DEV_GRAPHICS_PIDS
     *                          DCGM_FI_DEV_COMPUTE_PIDS
     * excludePid      IN: Optional PID to exclude from the list. 0=no exclusion
     * pids                 OUT: pidInfo, contains PID and the utilization rates.
     * numPids         IO: On calling, contains the capacity of pids. The output
     *                     is the number of entries set in pids
     * startTime       IN: Optional starting timestamp. Samples >= this value will
     *                     be returned. 0=Return samples from beginning
     * endTime         IN: Optional ending timestamp. Samples <= this value will
     *                     be returned. 0=Return samples up until end
     *
     * Returns: 0 on success.
     *          DCGM_ST_? enum on error.
     */
    dcgmReturn_t GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                   unsigned int excludePid,
                                   unsigned int *pids, unsigned int *numPids,
                                   timelib64_t startTime, timelib64_t endTime);
    

    /*************************************************************************/
    /*
     * Get a list of unique pid utilization entries in the utilization sample.
     *
     * entityGroupId   IN: Which entity group this entity belongs to
     * entityId        IN: Which entity to get process Utilization for
     * dcgmfieldId     IN: Field ID to get the process Utilization from. Lwrrently supported fields:
     *                          DCGM_FI_DEV_GPU_UTIL_SAMPLES
     *                          DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES
     * includePid      IN: (Optional)If specified, the utilization data
                             for only this PID is being retrieved.
     * processUtilSamples         OUT: Unique PID utilization samples.
     * numUniqueSamples        OUT: Number of Unique pid utilization samples.
     * startTime       IN: Optional starting timestamp. Samples >= this value will
     *                     be returned. 0=Return samples from beginning
     * endTime         IN: Optional ending timestamp. Samples <= this value will
     *                     be returned. 0=Return samples up until end
     *
     * Returns: 0 on success.
     *          DCGM_ST_? enum on error.
     */
    dcgmReturn_t GetUniquePidUtilLists(dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                                       unsigned int includePid, dcgmProcessUtilSample_t*processUtilSamples,
                                       unsigned int *numUniqueSamples,
                                       timelib64_t startTime, timelib64_t endTime);
                                                     

    /*************************************************************************/
    /*
     * Get summary information for a set of samples
     *
     * entityGroupId IN: Which entity group to get the value for
     * entityId      IN: The entity to get the value for
     * dcgmFieldId     IN: Which DCGM field to get the summaries for
     * numSummaryTypes IN: Number of entries in summaryTypes[] and summaryValues[]
     * summaryTypes    IN: Types of summary data to request
     * summaryValues  OUT: Summary values at hte same indexes as summaryTypes
     * startTime       IN: Optional starting timestamp. Samples >= this value will
     *                     be returned. 0=Return samples from beginning
     * endTime         IN: Optional ending timestamp. Samples <= this value will
     *                     be returned. 0=Return samples up until end
     * enumCB          IN: Callback to check if the entry should be used for callwlation of summary 
     * userData        IN: User provided data to be passed to the callback function
     *
     */
    dcgmReturn_t GetInt64SummaryData(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                     unsigned short dcgmFieldId, int numSummaryTypes,
                                     DcgmcmSummaryType_t *summaryTypes,
                                     long long *summaryValues,
                                     timelib64_t startTime, timelib64_t endTime, 
                                     pfUseEntryForSummary enumCB, void *userData);
    dcgmReturn_t GetFp64SummaryData(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                    unsigned short dcgmFieldId, int numSummaryTypes,
                                    DcgmcmSummaryType_t *summaryTypes,
                                    double *summaryValues,
                                    timelib64_t startTime, timelib64_t endTime, 
                                    pfUseEntryForSummary enumCB, void *userData);

    /*************************************************************************/
    /*
     * Get samples of a time series field
     *
     * entityGroupId IN: Which entity group to get the value for
     * entityId      IN: The entity to get the value for
     * dcgmFieldId  IN: Which DCGM field to get the value for
     * samples     OUT: Where to place samples. Capacity of this memory should
     *                  be provided in Msamples
     * Msamples     IO: When called, is the capacity that samples can hold.
     *                  When returning, Msamples contains the number of samples
     *                  actually stored in samples[]
     * sampleType  OUT: Type of samples returned in samples. This is a DCGM_FT_?
     *                  #define and redundant with the DcgmField definition
     * startTime    IN: Optional starting timestamp. Samples >= this value will
     *                  be returned. 0=Return samples from beginning
     * endTime      IN: Optional ending timestamp. Samples <= this value will
     *                  be returned. 0=Return samples up until end
     * order        IN: Which order to walk records in.
     *                  DCGM_ORDER_ASCENDING = increasing timestamps, starting from
     *                       startTime
     *                  DCGM_ORDER_DESCENDING = decreasing timestamps, starting from
     *                       endTime
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t GetSamples(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                            unsigned short dcgmFieldId,
                            dcgmcm_sample_p samples, int *Msamples,
                            timelib64_t startTime, timelib64_t endTime,
                            dcgmOrder_t order);


    /*************************************************************************/
    /*
     * Get the most recent sample of a field
     *
     * entityGroupId IN: Which entity group to get the value for
     * entityId      IN: The entity to get the value for
     * dcgmFieldId   IN: Which DCGM field to get the value for
     * sample       OUT: Where to place sample (optional)
     * fvBuffer     OUT: Alternative place to place sample (optional)
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t GetLatestSample(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                                 unsigned short dcgmFieldId, dcgmcm_sample_p sample,
                                 DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Get the most recent sample of multiple entities for multiple fields into
     * a fvBuffer
     * 
     * There is both a cached version and live version of this API
     *
     * entityList    IN: Entities to fetch the latest values for
     * fieldIds      IN: Field IDs to fetch for each entity
     * fvBuffer     OUT: Where to place samples. 
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */
    dcgmReturn_t GetMultipleLatestSamples(std::vector<dcgmGroupEntityPair_t> & entities, 
                                          std::vector<unsigned short> & fieldIds,
                                          DcgmFvBuffer *fvBuffer);
    dcgmReturn_t GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> & entities, 
                                              std::vector<unsigned short> & fieldIds,
                                              DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Set value for a field
     *
     * gpuId        IN: Which GPU to get the value for
     * dcgmFieldId  IN: Which DCGM field to get the value for
     * value        IN: Value to be set
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     */    
    dcgmReturn_t SetValue(int gpuId, unsigned short dcgmFieldId, dcgmcm_sample_p value);
    
    /*************************************************************************/
    /**
     * Enables Sync boost for a group of GPUs
     * @param gpuIdList
     * @return 
     */
    dcgmReturn_t AddSyncBoostGrp(unsigned int *gpuIdList, unsigned int length, 
                                 unsigned int *syncBoostId);
    
    /**
     * Disables Sync boost for a group of GPUs
     * @param syncBoostId
     * @return 
     */
    dcgmReturn_t RemoveSyncBoostGrp(unsigned int syncBoostId);

    /*****************************************************************************/
    /*
     * AppendSamples is called from Fabric Manager to add switch statistics and  
     * error notifications. It is implemented as a separate method from          
     * InjectSamples, in case the two functionalities need to diverge in future. 
     * For now, AppendSamples just ilwokes InjectSamples.                        
     *
     * entityGroupId IN: Which entity group to inject the value for
     * entityId      IN: ID of the entity to inject (GPU ID for GPUs)
     * dcgmFieldId   IN: Which DCGM field to inject the value for
     * samples       IN: Array of samples to inject
     * Nsamples      IN: Number of samples that are contained in samples[]
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     *
     */
    dcgmReturn_t AppendSamples(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               unsigned short dcgmFieldId,
                               dcgmcm_sample_p samples, int Nsamples);
    
    /*****************************************************************************/
    /*
     * AppendSamples is called from DCGM modules to batch-publish metrics into
     * the cache manager.
     *
     * fvBuffer IN: Buffer of samples to publish into the cache manager. This remains
     *              owned by the caller after this call. 
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     *
     */
    dcgmReturn_t AppendSamples(DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Inject fake value(s) for a GPU for a field into the cache manager
     *
     * entityGroupId IN: Which entity group to inject the value for
     * entityId      IN: ID of the entity to inject (GPU ID for GPUs)
     * dcgmFieldId   IN: Which DCGM field to inject the value for
     * samples       IN: Array of samples to inject
     * Nsamples      IN: Number of samples that are contained in samples[]
     *
     * Returns 0 on success
     *        <0 on error. See DCGM_ST_? #defines
     *
     */
    dcgmReturn_t InjectSamples(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               unsigned short dcgmFieldId,
                               dcgmcm_sample_p samples, int Nsamples);

    /*************************************************************************/
    /*
     * Free an array of DCGM samples, freeing any memory they might have
     * allocated
     *
     * samples      IN: Array of samples to free
     * Nsamples     IN: Number of samples that are contained in samples[]
     * dcgmFieldId  IN: Which DCGM field to free the value for. This is used to
     *                  get the field type.
     */
    dcgmReturn_t FreeSamples(dcgmcm_sample_p samples, int Nsamples,
                             unsigned short dcgmFieldId);

    /*************************************************************************/
    /*
     * Do the actual work of updating all fields, looking to see if the fields
     * should be updated. Call this from the DcgmCacheThread.
     *
     * threadCtx           IO: Update thread context
     * earliestNextUpdate OUT: Timestamp in usec since 1970 of the next time
     *                         any stat should be updated (minimum timestamp)
     *
     */
    dcgmReturn_t ActuallyUpdateAllFields(dcgmcm_update_thread_t *threadCtx, 
                                         timelib64_t *earliestNextUpdate);

    /*************************************************************************/
    /*
     * Populate a cache manager field info structure
     *
     * fieldInfo  IN/OUT: Structure to populate. fieldInfo->gpuId and fieldInfo->fieldId
     *                    are used to find the correct field record to query
     *
     *
     */
    dcgmReturn_t GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo);

    /*************************************************************************/
    /*
     * Populate a dcgmLwLinkStatus_v1 response with the LwLink link states
     * of every GPU and LwSwitch in the system
     */
    dcgmReturn_t PopulateLwLinkLinkStatus(dcgmLwLinkStatus_v1 &lwLinkStatus);

    /*************************************************************************/
    /*
     * Populate a dcgmLwLinkStatus_v2 response with the LwLink link states
     * of every GPU and LwSwitch in the system
     */
    dcgmReturn_t PopulateLwLinkLinkStatus(dcgmLwLinkStatus_v2 &lwLinkStatus);

    /*************************************************************************/
    /*
     * Get the state of LwLinks for a given entity. linkStates should be the following sizes:
     * DCGM_LWLINK_MAX_LINKS_PER_GPU for entityGroupId == DCGM_FE_GPU
     * DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH for entityGroupId == DCGM_FE_SWITCH
     * 
     */
    dcgmReturn_t GetEntityLwLinkLinkStatus(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           dcgmLwLinkLinkState_t *linkStates);

    /*************************************************************************/
    /*
     * Colwert a LWML return code to a LWML return code
     *
     * This method is static so it can be called from other modules
     *
     */
    static
    dcgmReturn_t LwmlReturnToDcgmReturn(lwmlReturn_t lwmlReturn);

    /*************************************************************************/
    /*
     * Colwert an LWML GPU LWLink error type to corresponding DCGM error type
     *
     */
    int LwmlGpuLWLinkErrorToDcgmError(long eventType);

    /*************************************************************************/
    /*
     * Map GPU ID to lwml index
     *
     * Returns LWML index on success. < 0 on failure (DCGMCM_ST_? #define)
     *
     */
    int GpuIdToLwmlIndex(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Map LWML index to gpuId
     *
     * Returns gpuId
     *
     */
    unsigned int LwmlIndexToGpuId(int lwmlIndex);

    /*************************************************************************/
    /* Colwert a LWML return code to an appropriate null value */
    static char *LwmlErrorToStringValue(lwmlReturn_t lwmlReturn);
    static long long LwmlErrorToInt64Value(lwmlReturn_t lwmlReturn);
    static int LwmlErrorToInt32Value(lwmlReturn_t lwmlReturn);
    static double LwmlErrorToDoubleValue(lwmlReturn_t lwmlReturn);

    /*************************************************************************/
    /*
     * Returns whether or not a GPU is whitelisted to run DCGM
     *
     */
    int IsGpuWhitelisted(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Returns whether a given entityId is valid or not
     */
    bool GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, 
                            dcgm_field_eid_t entityId);

    /*************************************************************************/
    /*
     * Add a fake, injection-only GPU to the cache manager and set the PCI
     * device id as well as the subsystem id
     *
     * Returns gpuId of the fake GPU that was created or DCGM_GPU_ID_BAD if one
     *         could not be created
     *
     */
    unsigned int AddFakeGpu(unsigned int pciDeviceId, unsigned int pciSubSystemId);

    /*************************************************************************/
    /*
     * Add a fake, injection-only GPU to the cache manager
     *
     * Returns gpuId of the fake GPU that was created or DCGM_GPU_ID_BAD if one
     *         could not be created
     *
     */
    unsigned int AddFakeGpu(void);

    /*************************************************************************/
    /*
     * Add a fake, injection-only LwSwitch to the cache manager
     *
     * Returns entityId of the fake lwSwitch that was created or DCGM_ENTITY_ID_BAD if one
     *         could not be created
     *
     */
    unsigned int AddFakeLwSwitch(void);

    /*************************************************************************/
    /*
     * Add a live LwSwitch to the cache manager
     *
     * physicalId IN: Physical switchId of the switch to add. This will become
     *                the entityId
     * 
     * Returns DCGM_ST_OK on success
     *         Other DCGM_ST_? error code on failure.
     *
     */
    dcgmReturn_t AddLwSwitch(unsigned int physicalId);

    /*************************************************************************/
    /*
     * Set the link state for a LwSwitch port
     *
     * physicalId IN: Physical switchId of the switch
     * portIndex  IN: LwSwitch port index to set the state of (0 to (DCGM_LWLINK_MAX_LINKS_PER_LWSWITCH-1))
     * linkState  IN: State of the link to set
     * 
     * Returns DCGM_ST_OK on success
     *         Other DCGM_ST_? error code on failure.
     *
     */
    dcgmReturn_t SetLwSwitchLinkState(unsigned int physicalId, unsigned int portIndex,
                                      dcgmLwLinkLinkState_t linkState);
    
    /*************************************************************************/
    /*
     * Set the link state for a GPU's LwLink link
     *
     * gpuId     IN: ID of the GPU
     * linkId    IN: Link index to set the state of (0 to (DCGM_LWLINK_MAX_LINKS_PER_GPU-1))
     * linkState IN: State of the link to set
     * 
     * Returns DCGM_ST_OK on success
     *         Other DCGM_ST_? error code on failure.
     *
     */
    dcgmReturn_t SetGpuLwLinkLinkState(unsigned int gpuId, unsigned int linkId,
                                       dcgmLwLinkLinkState_t linkState);


    /*************************************************************************/
    /*
     * Set the link state for an Entity's LwLink link
     *
     * entityGroupId IN: Entity group of the entity
     * entityId      IN: Entity ID of the entity
     * linkId        IN: Link index (portId for LwSwitches)
     * linkState     IN: State of the link to set
     * 
     * Returns DCGM_ST_OK on success
     *         Other DCGM_ST_? error code on failure.
     *
     */
    dcgmReturn_t SetEntityLwLinkLinkState(dcgm_field_entity_group_t entityGroupId, 
                                          dcgm_field_eid_t entityId,
                                          unsigned int linkId,
                                          dcgmLwLinkLinkState_t linkState);

    /*************************************************************************/
    /*
     * Are all of the GPUIDs in this group the same SKU?
     *
     * gpuIds IN: GPU Ids to compare for SKU equivalence. Note that this list
     *            is passed by reference, so make a copy of any thread-unsafe list
     *
     * Returns 1 if all of the GPUs of this group are the same
     *         0 if any of the GPUs of this group are different from each other
     *
     */
    int AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds);

    /*************************************************************************/
    /*
     * Return the approximate amount of memory, in bytes, used to store the given field ID.
     * This number is not meant to be precise but is meant to give a general idea
     * of the memory usage.
     *
     * This number will always be smaller than what is actually taken up.
     */
    dcgmReturn_t GetGpuFieldBytesUsed(unsigned int gpuId, unsigned short dcgmFieldId, long long *bytesUsed);

    /*************************************************************************/
    /*
     * Return the approximate amount of memory, in bytes, used to store the given field ID.
     * This number is not meant to be precise but is meant to give a general idea
     * of the memory usage.
     *
     * This number will always be smaller than what is actually taken up.
     */
    dcgmReturn_t GetGlobalFieldBytesUsed(unsigned short dcgmFieldId, long long *bytesUsed);

    /*************************************************************************/
    /*
     * Get the total amount of time, in usec, that the cache manager has spent retrieving
     * the given field on the given GPU since it started.
     *
     * totalUsec       OUT: the total time in usec
     */
    dcgmReturn_t GetGpuFieldExecTimeUsec(unsigned int gpuId,
                                         unsigned short dcgmFieldId,
                                         long long *totalUsec);

    /*************************************************************************/
    /*
     * Get the total amount of time, in usec, that the cache manager has spent retrieving
     * the given field since it started.
     *
     * totalUsec       OUT: the total time in usec
     */
    dcgmReturn_t GetGlobalFieldExecTimeUsec(unsigned short dcgmFieldId,
                                            long long *totalUsec);

    /*************************************************************************/
    /*
     * Get the total amount of times that the cache manager has fetched a new value
     * for this field on this gpu.
     *
     * fetchCount       OUT: the fetch count
     */
    dcgmReturn_t GetGpuFieldFetchCount(unsigned int gpuId,
                                       unsigned short dcgmFieldId,
                                       long long *fetchCount);

    /*************************************************************************/
    /*
     * Get the total amount of times that the cache manager has fetched a new value
     * for this field.
     *
     * fetchCount       OUT: the fetch count
     */
    dcgmReturn_t GetGlobalFieldFetchCount(unsigned short dcgmFieldId,
                                          long long *fetchCount);

    /*************************************************************************/
    /*
     * Get runtime stats for the cache manager
     *
     * stats   OUT: Runtime stats of the cache manager
     */
    void GetRuntimeStats(dcgmcm_runtime_stats_p stats);

    /*************************************************************************/
    /*
     * Add field watches for the given vGPU instance
     *
     */
    void WatchVgpuFields(lwmlVgpuInstance_t vgpuId);

    /*************************************************************************/
    /*
     * Remove all field watches and clear the cache for the given vGPU instance
     *
     */
    dcgmReturn_t UnwatchVgpuFields(lwmlVgpuInstance_t vgpuId);

    /*************************************************************************/
    /*
     * Manage the dynamic addition/removal of the vGPUs.
     * Active vGPU instance (linked) list is maintained per GPU within the function depending on addition of
     * new vGPU instance or removal of any running vGPU instance.
     * Also vgpuIndex is mapped to the vGPU instances running within range specified for the lwmlIndex.
     * First element of vgpuInstanceIds array passed as input must hold the count of vGPU instances running.
     *
     * Note that index 0 of vgpuInstanceIds is ignored
     *
     * Returns 0 if OK
     *        <0 DCGM_ST_? on module error
     */
    dcgmReturn_t ManageVgpuList(unsigned int gpuId, lwmlVgpuInstance_t *vgpuInstanceIds);

    /*************************************************************************/
    /*
     * Get the affinity information from LWML for this box
     */
    dcgmReturn_t PopulateTopologyAffinity(dcgmAffinity_t &affinity);

    /*************************************************************************/
    /*
     * Get and store the affinity information from LWML for this box
     */
    dcgmReturn_t CacheTopologyAffinity(dcgmcm_update_thread_t *threadCtx, 
                                       timelib64_t now, timelib64_t expireTime);

    /*************************************************************************/
    /*
     * From gpuIds, select numGpus of those which are the topologically best fit for each other, writing that 
     * information into outputGpus as a bitmask.
     *
     * Returns 0 if OK
     *        <0 DCGM_ST_* on module error
     */
    dcgmReturn_t SelectGpusByTopology(std::vector<unsigned int> &gpuIds, uint32_t numGpus, uint64_t &outputGpus);

    /*************************************************************************/
    /*
     * Set a bit in the bitmask for each gpu in the gpuIds vector
     */
    void ColwertVectorToBitmask(std::vector<unsigned int> &gpuIds, uint64_t &outputGpus, uint32_t numGpus);

    /*************************************************************************/
    /*
     * Fill the passed in affinity struct with the affinity information for this node.
     *
     * Returns 0 if OK
     *         DCGM_ST_* on module error
     */
    dcgmReturn_t PopulateCpuAffinity(dcgmAffinity_t &affinity);

    /*************************************************************************/
    /*
     * Add each GPU to a vector with every other GPU that shares it's cpu affinity.
     * Mose of the time there will be one or two groups.
     */
    void CreateGroupsFromCpuAffinities(dcgmAffinity_t &affinity,
                                       std::vector<std::vector<unsigned int> > &affinityGroups,
                                       std::vector<unsigned int> &gpuIds);
    
    /*************************************************************************/
    /*
     * Add the index (into affinityGroups) of each group large enough to meet this request
     */
    void PopulatePotentialCpuMatches(std::vector<std::vector<unsigned int> > &affinityGroups,
                                     std::vector<size_t> &potentialCpuMatches, uint32_t numGpus);

    /*************************************************************************/
    /*
     * Create a list of gpus from the groups to fulfill the request. This is only done if
     * no individual group of gpus (based on cpu affinity) could fulfill the request.
     */
    dcgmReturn_t CombineAffinityGroups(std::vector<std::vector<unsigned int> > &affinityGroups,
                                       std::vector<unsigned int> &combinedGpuList, int remaining);

    /*************************************************************************/
    /*
     * Return a topology struct populated with the information on the LWLink for
     * this system.
     *
     * Returns a pointer to the populated topology object
     *         NULL on error
     */
    dcgmTopology_t *GetLwLinkTopologyInformation();

    /*************************************************************************/
    /*
     * Choose the first grouping that has an ideal match based on LwLink topology.
     * This is only done if we have more than one group of GPUs that is ideal based
     * on CPU affinity.
     */
    void MatchByIO(std::vector<std::vector<unsigned int> > &affinityGroups, dcgmTopology_t *topPtr,
                   std::vector<size_t> &potentialCpuMatches, uint32_t numGpus, uint64_t &outputGpus);

    /*************************************************************************/
    /*
     * Record the number of connections this topology has between GPUs at each level, 0-3.
     * 0 is the fastest and 3 is the slowest.
     */
    unsigned int SetIOConnectionLevels(std::vector<unsigned int> &affinityGroup, dcgmTopology_t * topPtr,
                               std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > &connectionLevel);

    unsigned int RecordBestPath(std::vector<unsigned int> &bestPath,
                                std::map<unsigned int, std::vector<DcgmGpuConnectionPair> > &connectionLevel,
                                uint32_t numGpus, unsigned int highestLevel);

    /*************************************************************************/
    /*
     * Translate each path bitmap into the number of LwLinks that connect the two paths.
     */
    unsigned int LwLinkScore(dcgmGpuTopologyLevel_t path);

    /*************************************************************************/
    /*
     * Determines whether or not connections has numGpus total gpus that can all be linked together.
     * If there are, outputGpus is set with the gpus from the connection.
     *
     * Returns true if there are numGpus in the pairs inside connections that can be linked together.
     */
    bool HasStrongConnection(std::vector<DcgmGpuConnectionPair> &connections, uint32_t numGpus, uint64_t &outputGpus);

    /*************************************************************************/
    /*
     * Get the number of active LwLinks to LwSwitches per GPU. 
     * 
     * gpuCounts is an array of m_gpus to DCGM_MAX_NUM_DEVICES GPUs to hold the
     * gpu->LwSwitch counts for each GPU.
     * 
     * Will return 0 <= numLwLinks
     * Returns DCGM_ST_OK if gpuCounts are populated
     */
    dcgmReturn_t GetActiveLwSwitchLwLinkCountsForAllGpus(unsigned int *gpuCounts);

    /*************************************************************************/
    /*
     * Set topology_np to a pointer to a struct populated with the LwLink topology information
     *
     * Returns 0 if ok
     *         DCGM_ST_* on module error
     */
    dcgmReturn_t PopulateTopologyLwLink(dcgmTopology_t **topology_pp, unsigned int &topologySize);

    /*************************************************************************/
    /*
     * Check if the affinity bitmasks for the gpus at index1 and index2 match each other.
     *
     * Returns true if the gpus have the same CPU affinity
     *         false if not
     */
    bool AffinityBitmasksMatch(dcgmAffinity_t &affinity, unsigned int index1, unsigned int index2);

    /*************************************************************************/
    /*
     * Cache the topology information relevant to LwLink for this host.
     *
     * Returns 0 if ok
     *         DCGM_ST_* on module error
     */
    dcgmReturn_t CacheTopologyLwLink(dcgmcm_update_thread_t *threadCtx,
                                     timelib64_t now, timelib64_t expireTime);

    /*************************************************************************/
    /*
     * Get the cache manager's list of valid field IDs.
     * 
     * includeModulePublished IN: Should we include fields that are module-published in this list?
     *
     */
    void GetValidFieldIds(std::vector<unsigned short> &validFieldIds, bool includeModulePublished=true);

    /*************************************************************************/
    /*
     * Get a copy of the watch info for a given entity for a given field into watchInfo.
     * This is useful for testing purposes. This information is a snapshot of what the cache manager
     * has internally for the entity.
     *
     * Returns: DCGM_ST_OK on success.
     *          DCGM_ST_NO_DATA if the watch object cannot be found.
     *
     */
    dcgmReturn_t GetEntityWatchInfoSnapshot(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            unsigned int fieldId, dcgmcm_watch_info_p watchInfo);
    
    /*************************************************************************/
    /*
     * Handle a client disconnecting from the host engine
     */
    void OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);
    
    /*************************************************************************/
    /*
     * Main loop thread for the event reading thread this is public so it can
     * be seen from DcgmCacheManagerEventThread. This method should not be called
     * externally.
     * 
     */
    void EventThreadMain(DcgmCacheManagerEventThread *eventThread);

    /*************************************************************************/
    /*
     * Find all GPUs in the system and set their state appropriately in this
     * object.
     *
     * RETURNS: DCGM_ST_OK on success
     *          DCGM_ST_GENERIC_ERROR on LWML error
     */
    dcgmReturn_t AttachGpus(void);
    
    /*************************************************************************/
    /*
     * Adds the newly detected GPU list to our internal list by matching UUIDs
     */
    void MergeNewlyDetectedGpuList(dcgmcm_gpu_info_p detectedGpus, int count);

    /*************************************************************************/
    /*
     * Detach from all GPUs in the system and set their state appropriately.
     *
     * RETURNS: DCGM_ST_OK on success
     *          DCGM_ST_GENERIC_ERROR on LWML error
     */
    dcgmReturn_t DetachGpus(void);

    /*************************************************************************/
    /*
     * Gets the index of the GPU with a matching gpuId from m_gpus.
     *
     * RETURNS: the index if found
     *          -1 if no matching GPU was found
     */
    int FindGpuIdsIndex(unsigned int gpuId);

    /*************************************************************************/

private:

    int m_pollInLockStep;        /* Whether to poll when told to (1) or at the
                                    frequency of the most frequent stat being tracked (0) */

    timelib64_t m_maxSampleAgeUsec; /* Maximum time to keep a sample of ANY field, defaults to 1 hour */

    unsigned int m_numGpus; /* Number of entries in m_gpus[] that are valid */
    dcgmcm_gpu_info_t m_gpus[DCGM_MAX_NUM_DEVICES]; /* All of the GPUs we know about, indexed by gpuId */

    bool m_lwmlInitted; /* Tracks whether or not LWML has been initialized. since LWML keeps a count
                           of initializations, we don't want to double initialize or we won't be able
                           to detach and attach to GPUs correctly. */

    int m_numLwSwitches; /* Number of entries in m_lwSwitches that are valid */
    dcgmcm_lwswitch_info_t m_lwSwitches[DCGM_MAX_NUM_SWITCHES]; /* All of the LwSwitches we know about */

    std::vector<lwmlBlacklistDeviceInfo_t> m_gpuBlacklist; /* Array of GPUs that have been blacklisted by the driver */

    DcgmMutex *m_mutex; /* Lock used for protecting data structures within this class */
    unsigned int m_inDriverCount; // Count of threads lwrrently in driver calls
    unsigned int m_waitForDriverClearCount; // Count of threads waiting for the driver to be clear 

    lwosCV m_startUpdateCondition; /* Condition used for signaling the update thread to wake up.
                                      It should be passed with m_lock to lwosCondWait */
    lwosCV m_updateCompleteCondition; /* Condition used for telling waiting threads that an
                                         update loop has completed */

    /* Track per-entity watches of fields. Use GetEntityWatchInfo() method to get a 
       pointer to an element of this. 
       Is a hash of dcgmcm_entity_key_t -> entity_watch_table_t */
    hashtable_t *m_entityWatchHashTable;

    /* LWML internal callback table */
    etblLWMLCommonInternal *m_etblLwmlCommonInternal;

    /* Cache of which PIDs we have already saved to the cache with which start times
     * This saves us having to scan the entire accounting data structure to find
     * which PIDs we have already saved */
    keyedvector_p m_accountingPidsSeen;

    /* LWML events */

    /* Mask of current events being monitored. See lwmlEventType* in lwml.h */
    unsigned long long m_lwrrentEventMask[DCGM_MAX_NUM_DEVICES];
    bool m_lwmlEventSetInitialized;         /* Is m_lwmlEventSet initialized */
    lwmlEventSet_t m_lwmlEventSet;

    /* Vector of the field IDs that are actually defined. This will never change
     * after start-up, so iterators for this can be used without locking */
    std::vector<unsigned short>m_allValidFieldIds;

    /* Runtime stats of the cache manager */
    dcgmcm_runtime_stats_t m_runStats;

    DcgmCacheManagerEventThread *m_eventThread; /* Thread for reading LWML events */

    bool m_haveAnyLiveSubscribers; /* Has any watch registered to receive live updates? */

    std::vector<dcgmSubscribedFvUpdateCBEntry_t>m_onFvUpdateCBs; /* Callbacks to ilwoke when a 
                                                                    subscribed field value updates */

    /*************************************************************************/
    /*
     * Look at watchInfo's watchers and see if any have live subscriptions
     *
     * If any do, mark them in watchInfo->bufferedWatchers
     */
    void MarkSubscribersInThreadCtx(dcgmcm_update_thread_t *threadCtx, 
                                    dcgmcm_watch_info_p watchInfo);

    /*************************************************************************/
    /*
     * Call callbacks for any FVs that have been buffered by this update thread
     *
     */
    dcgmReturn_t UpdateFvSubscribers(dcgmcm_update_thread_t *updateCtx);

    /*************************************************************************/
    /*
     * Callwlate the max age in usec from the watch parameters
     *
     * The minimum (most restrictive) quota is used
     *
     * RETURNS: The value that should be used for maxAgeUsec
     */
    timelib64_t GetMaxAgeUsec(timelib64_t monitorFrequencyUsec, double maxAgeSeconds, int maxKeepSamples);

    /*************************************************************************/
    /*
     * Do some common pre-checks of a watchInfo before samples are processed for it
     * and return common error codes if any common conditions are encountered. 
     * 
     * Note: This code assumes that the cache manager is locked
     * 
     * Returns: DCGM_ST_OK if samples can be returned from this watchInfo
     *          Other DCGM_ST_? status codes that should be returned to the caller on error.
     */
    dcgmReturn_t PrecheckWatchInfoForSamples(dcgmcm_watch_info_p watchInfo);

    /*************************************************************************/
    /*
     * Helper function to enforce the quota for a watchInfo's time series.
     * 
     * watchInfo           IO: Watch info to check + modify
     * timestamp           IN: Current timestamp
     * oldestKeepTimestamp IN: Oldest timestamp to leave in the time series.
     *                         Any records with timestamps older than this
     *                         will be removed.
     */
    dcgmReturn_t EnforceWatchInfoQuota(dcgmcm_watch_info_p watchInfo, 
                                       timelib64_t timestamp, 
                                       timelib64_t oldestKeepTimestamp);

    /*************************************************************************/
    /*
     * Helper functions for adding or removing watch info classes
     */
    dcgmcm_watch_info_p AllocWatchInfo(dcgmcm_entity_key_t entityKey);
    void FreeWatchInfo(dcgmcm_watch_info_p watchInfo);

    /*************************************************************************/
    /*
     * Allocate the timeSeries part of a watchInfo
     * 
     * Returns: DCGM_ST_OK on success
     *          Any other DCGM_ST_? on error
     */
    dcgmReturn_t AllocWatchInfoTimeSeries(dcgmcm_watch_info_p watchInfo, int tsType);

    /*************************************************************************/
    /*
     * Add a watcher on a field or update the existing watcher if newWatcher is
     * already in the list
     *
     * NOTE: This function assumes the cache manager is already locked so that
     *       watchInfo is safe to modify
     *
     * RETURNS: Always returns DCGM_ST_OK for now
     *
     */
    dcgmReturn_t AddOrUpdateWatcher(dcgmcm_watch_info_p watchInfo,
                                    bool *wasAdded,
                                    dcgm_watch_watcher_info_t *newWatcher);
    
    /*************************************************************************/
    /*
     * Remove a watcher on a field
     *
     * NOTE: This function assumes the cache manager is already locked so that
     *       watchInfo is safe to modify
     *
     * RETURNS: DCGM_ST_OK if the watcher was found and removed
     *          DCGM_ST_NOT_WATCHED if the given watcher wasn't found
     *
     */
    dcgmReturn_t RemoveWatcher(dcgmcm_watch_info_p watchInfo,
                               dcgm_watch_watcher_info_t *watcher,
                               int clearCache);
    
    /*************************************************************************/
    /*
     * Update the update update frequency and quota based on the minimum values
     * from all of our watchers
     *
     * NOTE: This function assumes the cache manager is already locked so that
     *       watchInfo is safe to modify
     */
    dcgmReturn_t UpdateWatchFromWatchers(dcgmcm_watch_info_p watchInfo);

    /*************************************************************************/
    /*
     * Tell the cache manager to update its LwLink link state for a given gpuId
     *
     */
    dcgmReturn_t UpdateLwLinkLinkState(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Colwert a entityId and fieldId to a watch key to be used as an index
     * against m_entityWatches.
     *
     * Returns: !0 watch key on success
     *           0 on error
     *
     */
    void EntityIdToWatchKey(dcgmcm_entity_key_t *watchKey,
                            dcgm_field_entity_group_t entityGroupId,
                            dcgm_field_eid_t entityId,
                            unsigned int fieldId);

    /*************************************************************************/
    /*
     * Get the watch context structure for a field, given it's gpuId and fieldId
     *
     * entityGroupId: ID of the entity group
     * entityId: ID of the entity within the group. This should be gpuId for GPUs and VGPU ID for VGPUs
     * fieldId: fieldId of the field
     * createIfNotExists: If the watchInfo doesn't already exist, should we create it? 1=yes. 0=no.
     *
     * Returns: pointer to watch context structure
     *          NULL on error (likely out of memory)
     *
     */
    dcgmcm_watch_info_p GetEntityWatchInfo(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           unsigned int fieldId, int createIfNotExists);
    dcgmcm_watch_info_p GetGlobalWatchInfo(unsigned int fieldId, int createIfNotExists);

    /*************************************************************************/
    /*
     * Get the next watch_info structure that is valid for a given lwmlIndex
     *
     * This method is useful for iterating over all watched fields of a GPU
     *
     * Pass *validFieldIndex as 0 on the first call
     *
     * RETURNS: Pointer to watch info structure on success
     *          NULL if we've reached the end of the watches for this lwmlIndex
     *
     */
    dcgmcm_watch_info_p GetNextEntityWatchInfo(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned int *validFieldIndex);

    /*************************************************************************/
    /*
     * Initialize and clear a thread context variable so it's ready for next use.
     * 
     * This should do as few memsets as possible as this is in the critical path
     * for module-published field-value processing.
     */
    void InitAndClearThreadCtx(dcgmcm_update_thread_t *threadCtx);

    /*************************************************************************/
    /*
     * Clear a thread context variable so it's ready for next use.
     * 
     * This should do as few memsets as possible as this is in the critical path
     * for live field-value processing.
     */
    void ClearThreadCtx(dcgmcm_update_thread_t *threadCtx);

    /*************************************************************************/
    /*
     * Free the contents of a thread context variable. This does not free threadCtx
     * itself in case it's part of another structure or on the stack
     * 
     */
    void FreeThreadCtx(dcgmcm_update_thread_t *threadCtx);

    /*************************************************************************/
    /*
     * Check to see if an accounting PID has already been cached at a given timestamp
     *
     * RETURNS: 1 if PID has been cached
     *          0 if not
     */
    int HasAccountingPidBeenSeen(unsigned int pid, timelib64_t timestamp);

    /*************************************************************************/
    /*
     * Mark a PID as having been cached
     *
     * RETURNS: 0 if OK
     *         <0 on error. See DCGM_ST_? enumerations for details.
     *
     */
    dcgmReturn_t CacheAccountingPid(unsigned int pid, timelib64_t timestamp);

    /*************************************************************************/
    /*
     * Clear all values from the accounting PID cache
     *
     * RETURNS: Nothing
     *
     */
    void EmptyAccountingPidCache();

    /*************************************************************************/
    /*
     * Read a value for a GPU field and cache or buffer it. See
     * dcgmcm_update_thread_t documentation for details.
     *
     * Returns 0 if OK
     *        <0 DCGM_ST_? on module error
     */
    dcgmReturn_t BufferOrCacheLatestGpuValue(dcgmcm_update_thread_t *threadCtx,
                                             dcgm_field_meta_p fieldMeta);

    /*************************************************************************/
    /*
     * Cache or buffer the latest value for a watched vGPU field
     *
     * Returns 0 if OK
     *        <0 DCGM_ST_? on module error
     */
    dcgmReturn_t BufferOrCacheLatestVgpuValue(dcgmcm_update_thread_t *threadCtx, 
                                              lwmlVgpuInstance_t vgpuId, 
                                              dcgm_field_meta_p fieldMeta);

    /*************************************************************************/
    /*
     * Helper method to add entity field watches
     */
    dcgmReturn_t AddEntityFieldWatch(dcgm_field_entity_group_t fieldEntityGroup, unsigned int entityId,
                                     unsigned short dcgmFieldId,
                                     timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                     int maxKeepSamples, DcgmWatcher watcher, bool subscribeForUpdates);

    /*************************************************************************/
    /*
     * Helper method to remove device field watches
     */
    dcgmReturn_t RemoveEntityFieldWatch(dcgm_field_entity_group_t fieldEntityGroup, unsigned int entityId,
                                        unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Clear the contents of a watch object in the cache manager
     *
     * clearCache    IN: Whether to clear cached data for this entity. 1=yes. 0=no 
     *
     * NOTE: Assumes the cache manager is locked by the caller
     */
    void ClearWatchInfo(dcgmcm_watch_info_p watchInfo, int clearCache);
    

    /*************************************************************************/
    /*
     * Clear all entities from the cache manager. The watch objects will remain
     * but their contents will be cleared
     *
     * clearCache    IN: Whether to clear cached data for this entity. 1=yes. 0=no
     */
    dcgmReturn_t ClearAllEntities(int clearCache);
    
    /*************************************************************************/
    /*
     * Clear an entity from the cache manager. The watch object will remain
     * but its contents will be cleared
     *
     * entityGroupId IN: Group id of the entity to clear
     * entityId      IN: Entity within entityGroupId to clear
     * clearCache    IN: Whether to clear cached data for this entity. 1=yes. 0=no
     *
     */
    dcgmReturn_t ClearEntity(dcgm_field_entity_group_t entityGroupId,
                             dcgm_field_eid_t entityId,
                             int clearCache);

    /*************************************************************************/
    /*
     * Helper method to add global watches
     */
    dcgmReturn_t AddGlobalFieldWatch(unsigned short dcgmFieldId, timelib64_t monitorFrequencyUsec,
                               double maxSampleAge, int maxKeepSamples, DcgmWatcher watcher,
                               bool subscribeForUpdates);

    /*************************************************************************/
    /*
     * Helper method to Stop and remove monitoring of a global field
     *
     */
    dcgmReturn_t RemoveGlobalFieldWatch(unsigned short dcgmFieldId, int clearCache, 
                                        DcgmWatcher watcher);
    
    /*************************************************************************/
    /*
     * Helper method to return whether (true) or not (false) a fieldId is published
     * by a module. These fieldIds will not be updated by the DCGM cache manager.
     *
     */
    bool IsModulePushedFieldId(unsigned int fieldId);

    /*************************************************************************/
    /*
     * Helper functions to append values to our internal data structure
     * 
     * Note that threadCtx has entries that should be set before this call:
     * - entityKey MUST be set to the key information for this update
     * - fvBuffer can be set if you want this value to be buffered. This is useful
     *       for capturing updates and sending them to other subsystems or clients
     * - watchInfo can be set if you want this value to be cached    
     * 
     * Returns 0 on success
     *         DCGM_ST_? #define on error
     *
     */
    dcgmReturn_t AppendEntityString(dcgmcm_update_thread_t *threadCtx, char *value, 
                                    timelib64_t timestamp, timelib64_t oldestKeepTimestamp);
    dcgmReturn_t AppendEntityDouble(dcgmcm_update_thread_t *threadCtx, double value1, double value2,
                                    timelib64_t timestamp, timelib64_t oldestKeepTimestamp);
    dcgmReturn_t AppendEntityInt64(dcgmcm_update_thread_t *threadCtx,
                                   long long value1, long long value2, 
                                   timelib64_t timestamp, timelib64_t oldestKeepTimestamp);
    dcgmReturn_t AppendEntityBlob(dcgmcm_update_thread_t *threadCtx, void *value, int valueSize,
                                  timelib64_t timestamp, timelib64_t oldestKeepTimestamp);

    dcgmReturn_t AppendDeviceSupportedClocks(dcgmcm_update_thread_t *threadCtx,
                                             lwmlSupportedClocks_t *supportedClocks,
                                             timelib64_t timestamp, timelib64_t oldestKeepTimestamp);
    
    dcgmReturn_t AppendSyncBoostGroups(dcgmcm_update_thread_t *threadCtx, lwmlSyncBoostGroupList_t *syncBoostList,
                                       timelib64_t timestamp, timelib64_t oldestKeepTimestamp);

    dcgmReturn_t AppendDeviceAccountingStats(dcgmcm_update_thread_t *threadCtx, unsigned int pid,
                                             lwmlAccountingStats_t *lwmlAccountingStats,
                                             timelib64_t timestamp, timelib64_t oldestKeepTimestamp);

    /*************************************************************************/
    /*
     * Helper to update all fields for a GPU that can be retrieved in batch
     * from LWML via their lwmlFieldId.
     *
     * threadCtx     IN: Thread context that has the following entries populated:
     *                   numFieldIds[gpuId] -> Number of entries in next two
     *                   fieldValueFields[gpuId] -> Field meta for each field
     *                   fieldValueWatchInfo[gpuId] -> Watch info for each field. NULL=don't cache (just buffer)
     * gpuId         IN: ID of the GPU to fetch fields for
     *
     */
    dcgmReturn_t ActuallyUpdateGpuFieldValues(dcgmcm_update_thread_t *threadCtx,
                                              unsigned int gpuId);

    /*************************************************************************/
    /*
     * Polling thread run() top level helpers for either running in lock step
     * or timed mode
     *
     */
    void RunTimedWakeup(dcgmcm_update_thread_t *threadCtx);
    void RunLockStep(dcgmcm_update_thread_t *threadCtx);

    /*************************************************************************/
    /*
     * Helpers to prepare or unprepare LWML for field updates for a given field ID.
     * This handles tasks like telling LWML to watch/unwatch accounting data
     *
     * NOTE: This function assumes it is inside of a Lock() / Unlock() pair.
     *
     */
    dcgmReturn_t LwmlPreWatch(unsigned int gpuId, unsigned short dcgmFieldId);
    dcgmReturn_t LwmlPreWatchLwLink(unsigned int gpuId, unsigned int link);
    dcgmReturn_t LwmlPostWatch(unsigned int gpuId, unsigned short dcgmFieldId);

    /*************************************************************************/
    /*
     * Helper method to manage watching LWML events. Since events are read
     * system wide, we need to look at all GPUs' watches to see which events
     * we care about
     *
     * NOTE: This function assumes it is inside of a Lock() / Unlock() pair.
     *       Lwrrently, it is only called from LwmlPreWatch/LwmlPostWatch
     *
     * addWatchOnGpuId     IN: Optional GPU ID of a watch that will be created
     *                         after this call that we should prepare for. -1 = none
     * addWatchOnFieldId   IN: Optional DCGM field id of a watch that will be created
     *                         after this call that we should prepare for. 0 = none
     *
     *         Returns 0 on success
     *         DCGM_ST_? #define on error
     */
    dcgmReturn_t ManageDeviceEvents(unsigned int addWatchOnGpuId,
                                    unsigned short addWatchOnFieldId);

    /*
     * Check that the given field and gpuId are valid.
     */
    dcgmReturn_t CheckValidGlobalField(unsigned short dcgmFieldId);
    dcgmReturn_t CheckValidGpuField(unsigned int gpuId, unsigned short dcgmFieldId);

    /*************************************************************************/
    /*
     * Read the GPU blacklist from the driver an cache it into m_gpuBlacklist
     *
     */
    dcgmReturn_t ReadAndCacheGpuBlacklist(void);

    /*************************************************************************/
    /*
     * Helpers to fetch the information of active FBC sessions on the given device/vGPU instance
     *
     */
    dcgmReturn_t GetDeviceFBCSessionsInfo(lwmlDevice_t lwmlDevice, dcgmcm_update_thread_t *threadCtx,
                                          dcgmcm_watch_info_p watchInfo, timelib64_t now,
                                          timelib64_t expireTime);
    dcgmReturn_t GetVgpuInstanceFBCSessionsInfo(lwmlVgpuInstance_t vgpuId, dcgmcm_update_thread_t *threadCtx,
                                                dcgmcm_watch_info_p watchInfo, timelib64_t now,
                                                timelib64_t expireTime);

    /*************************************************************************/
    /*
     * Signifies a thread has entered the driver
     * NOTE: m_mutex must be locked before calling
     *
     * Checked by attaching and detaching to make sure unloading and loading LWML is ok
     */
    void MarkEnteredDriver();
    
    /*************************************************************************/
    /*
     * Signifies a thread has returned from the driver
     * NOTE: m_mutex must be locked before calling
     *
     * Checked by attaching and detaching to make sure unloading and loading LWML is ok
     */
    void MarkReturnedFromDriver();
    
    /*************************************************************************/
    /*
     * Tracks whether it's okay to enter the driver or proceed as though no one 
     * is in the driver.
     *
     * @see AttachGpus(), DetachGpus(), MarkEnteredDriver()
     */
    void SynchronizeDriverEntries(unsigned int &countToWaitFor, unsigned int &waitingEntries, bool entering);

    /*****************************************************************************/
    /*
     * Simple wrapper for SynchronizeDriverEntries(); waits for nothing to be
     * using the driver
     */
    void WaitForThreadsToExitDriver();

    /*****************************************************************************/
    /*
     * Simple wrapper for SynchronizeDriverEntries(); waits for the driver to be
     * okay to use again after detaching
     */
    void WaitForDriverToBeReady();
    
    /*************************************************************************/
    /*
     * Inserts an LWML error value translated to the relevant field type.
     */
    void InsertLwmlErrorValue(dcgmcm_update_thread_t *threadCtx, unsigned char fieldType, lwmlReturn_t err,
                          timelib64_t expireTime);

    /*************************************************************************/
    /*
     * Frees up the lwml event set and updates the internal tracking data
     */
    void UninitializeLwmlEventSet();

    /*************************************************************************/
    /*
     * Initializes the lwml event set and updates the internal tracking data
     */
    dcgmReturn_t InitializeLwmlEventSet();

    /*************************************************************************/
};

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedGpuField(Fn *callbackFn,
                                                      std::vector<unsigned short>* fieldIds,
                                                      unsigned int * const onlyOnThisGpuId)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedGpuField(callbackFns, fieldIds, onlyOnThisGpuId);
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedGpuField(std::vector<Fn *> callbackFns,
                                                      std::vector<unsigned short>* fieldIds,
                                                      unsigned int * const onlyOnThisGpuId)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    if (NULL == fieldIds)
    {
        fieldIds = &m_allValidFieldIds;
    }

    std::vector<unsigned int> gpuIds;

    if (onlyOnThisGpuId)
    {
        gpuIds.push_back(*onlyOnThisGpuId);
    }
    else
    {
        st = GetGpuIds(1, gpuIds);
        if (DCGM_ST_OK != st)
            return st;
    }

    for (std::vector<unsigned short>::iterator fieldId = fieldIds->begin();
         fieldId != fieldIds->end();
         ++fieldId)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(*fieldId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN || fieldMeta->scope != DCGM_FS_DEVICE)
        {
            continue;
        }

        for (std::vector<unsigned int>::iterator gpuId = gpuIds.begin(); gpuId != gpuIds.end(); ++gpuId)
        {
            watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, *gpuId, *fieldId, 0);
            if (!watchInfo || !watchInfo->isWatched)
            {
                continue;
            }

            typename std::vector<Fn *>::iterator it;
            for (it = callbackFns.begin(); it != callbackFns.end(); ++it)
            {
                Fn *fn = *it;
                st = (*fn)(*fieldId, *gpuId);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                }
            }
        }
    }

    return retSt;
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedGlobalField(Fn *callbackFn,
                                                         std::vector<unsigned short>* fieldIds)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedGlobalField(callbackFns, fieldIds);
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedGlobalField(std::vector<Fn *> callbackFns,
                                                         std::vector<unsigned short>* fieldIds)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    if (NULL == fieldIds)
    {
        fieldIds = &m_allValidFieldIds;
    }

    for (std::vector<unsigned short>::iterator fieldId = fieldIds->begin();
         fieldId != fieldIds->end();
         ++fieldId)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(*fieldId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN || fieldMeta->scope != DCGM_FS_GLOBAL)
        {
            continue;
        }

        watchInfo = GetGlobalWatchInfo(*fieldId, 0);
        if (!watchInfo || !watchInfo->isWatched)
        {
            continue;
        }

        typename std::vector<Fn *>::iterator it;
        for (it = callbackFns.begin(); it != callbackFns.end(); ++it)
        {
            Fn *fn = *it;
            st = (*fn)(*fieldId);
            if (DCGM_ST_OK != st)
            {
                retSt = st;
            }
        }
    }

    return retSt;
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedField(Fn *callbackFn,
                                                   std::vector<unsigned short>* fieldIds)
{
    std::vector<Fn *> callbackFns;
    callbackFns.push_back(callbackFn);
    return ForEachWatchedField(callbackFns, fieldIds);
}

// must be in header because of template
template <typename Fn>
dcgmReturn_t DcgmCacheManager::ForEachWatchedField(std::vector<Fn *> callbackFns,
                                                   std::vector<unsigned short>* fieldIds)
{
    dcgmReturn_t st;
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    if (NULL == fieldIds)
    {
        fieldIds = &m_allValidFieldIds;
    }

    std::vector<unsigned int> gpuIds;
    st = GetGpuIds(1, gpuIds);
    if (DCGM_ST_OK != st)
        return st;

    for (std::vector<unsigned short>::iterator fieldId = fieldIds->begin();
         fieldId != fieldIds->end();
         ++fieldId)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(*fieldId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope == DCGM_FS_DEVICE)
        {
            for (std::vector<unsigned int>::iterator gpuId = gpuIds.begin(); gpuId != gpuIds.end(); ++gpuId)
            {
                watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, *gpuId, *fieldId, 0);
                if (watchInfo && watchInfo->isWatched)
                {
                    typename std::vector<Fn *>::iterator it;
                    for (it = callbackFns.begin(); it != callbackFns.end(); ++it)
                    {
                        Fn *fn = *it;
                        st = (*fn)(*fieldId);
                        if (DCGM_ST_OK != st)
                        {
                            retSt = st;
                        }
                    }
                    break; // only run once even if watched on multiple GPUs
                }
            }
        }
        else if (fieldMeta->scope == DCGM_FS_GLOBAL)
        {
            watchInfo = GetGlobalWatchInfo(*fieldId, 0);
            if (watchInfo && watchInfo->isWatched)
            {
                typename std::vector<Fn *>::iterator it;
                for (it = callbackFns.begin(); it != callbackFns.end(); ++it)
                {
                    Fn *fn = *it;
                    st = (*fn)(*fieldId);
                    if (DCGM_ST_OK != st)
                    {
                        retSt = st;
                    }
                }
            }
        }
    }

    return retSt;
}

#endif /* DCGMCACHEMANAGER_H */
