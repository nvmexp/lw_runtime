#pragma once

#include "DcgmModule.h"
#include "dcgm_core_structs.h"
#include "dcgm_profiling_structs.h"
#include "LwcmCacheManager.h"
#include "LwcmGroup.h"
#include "DcgmTaskRunner.h"
#include "DcgmLopGpu.h"
#include <vector>

/* Global state for this module */
typedef enum
{
    DcgmProfStateUninitialized = 0,      /* Haven't initialized yet */
    DcgmProfStateInitialized,            /* Module is initialized but no session or config is active */
    DcgmProfStateUnknownError            /* Profiling is inactive due to an unknown error */
} dcgmProfGpuState_t;

/* Indicies into DcgmModuleProfiling::m_gpus[x].lopMetrics */
typedef enum
{
    dcgmProfMetricGrActive = 0,
    dcgmProfMetricSmActive,
    dcgmProfMetricSmOclwpancy,
    dcgmProfMetricPipeTensorActive,
    dcgmProfMetricDramActive,
    dcgmProfMetricPipeFp64Active,
    dcgmProfMetricPipeFp32Active,
    dcgmProfMetricPipeFp16Active,
    dcgmProfMetricPcieTxBytes,
    dcgmProfMetricPcieRxBytes,
    dcgmProfMetricLwLinkTxBytes,
    dcgmProfMetricLwLinkRxBytes,
    
    dcgmProfMetricCount /* Last entry */
} dcgmProfMetricIndex_t; 

/* Metadata about each metric for a given GPU */
typedef struct 
{
    dcgmProfMetricIndex_t id; /* ID of this event in DCGM. This is also the index 
                                 of this entry into the table it's stored in */
    bool isSupported;         /* Is this LOP metric supported for this GPU? */
    const char *lopTag;       /* String that this metric is identified by in LOP */
    double metricDivisor;     /* If != 0.0 or 1.0, this is what we should divide any value
                                 by before storing it in the DCGM cache */
    double minAllowedValue;   /* Minimum valid value for this metric */
    double maxAllowedValue;   /* Maximum valid value for this metric */
    std::vector<double> values; /* Aclwmulated values since the last time this event's results
                                 were snapshotted to the cache manager. This may be a single
                                 value or an array of values in the case of multi-pass metrics */
    unsigned short mgMajorId; /* Metric group major ID */
    unsigned short mgMinorId; /* Metric group minor ID */
} dcgmProfMetric_t;

/* Struct to represent a GPU in this module */
typedef struct dcgm_module_prof_gpu_t
{
    unsigned int gpuId;  /* The cache-manager assigned ID of this GPU */
    dcgmProfGpuState_t state; /* State of profiling on this GPU */
    lwmlChipArchitecture_t arch; /* chip architecture */
    dcgmGpuBrandType_t brand;    /* Brand */
    double memClockMhz;  /* Memory clock in MHz */
    double maximumMemBandwidth; /* Maximum memory bandwidth per second in bytes */
    dcgmProfMetric_t lopMetrics[dcgmProfMetricCount]; /* State of each of the LOP 
                                                         metrics for this GPU */
    timelib64_t lastEventQueryUsec; /* Last time updated this GPU's metrics from LOP */
    timelib64_t eventQueryIntervalUsec; /* How often we should update events of this GPU from LOP */
    timelib64_t lastCacheUsec; /* Last time we sent values to the cache manager in usec */
    
    std::vector<unsigned short> enabledFieldIds; /* FieldIDs that are lwrrently watched */
    std::map<unsigned int, dcgmProfMetricIndex_t> fieldIdsToMetrics; /* Table of 
                                    supported fieldIds and their underlying metrics */
    dcgmProfGetMetricGroups_t metricGroups; /* Supported metric groups for this GPU */
    DcgmLopGpu *lopGpu; /* object used for watching and retrieving metrics from LOP */

    std::vector<DcgmLopSample_t>sampleBuffer; /* Buffer used to read samples from lopGpu->GetSamples() */

    unsigned short smMajorGroup; /* Which major group is SM fieldIds. This is an optimization for
                                    multi-pass detection */
} dcgm_module_prof_gpu_t;

/* Structure to represent stats for this module */
typedef struct dcgm_module_prof_stats_t
{
    uint64_t totalMetricsRead;   /* Total count of metrics read from LOP (numCalls * metricsReturnedPerCall) */
    uint64_t numLopReadCalls;    /* Number of times we have read metrics from lop */
    timelib64_t totalUsecInLopReads; /* Total amount of time spent in LOP reads in usec */
} dcgm_module_prof_stats_t;

class DcgmModuleProfiling : public DcgmModule, DcgmTaskRunner
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleProfiling();
    virtual ~DcgmModuleProfiling(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     * (inherited from DcgmModule)
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * Process a DCGM module message from our taskrunner thread.
     */
    dcgmReturn_t ProcessMessageFromTaskRunner(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /* 
     * Process an entity group being destroyed (inherited from DcgmModule)
     */
    void OnGroupRemove(unsigned int groupId);

    /*************************************************************************/
    /* 
     * Process a client disconnecting (inherited from DcgmModule)
     */
    void OnClientDisconnect(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * This is the main worker function of the module. This is inherited from
     * DcgmTaskThread.
     * 
     * Returns: Minimum ms before we should call this function again. This will
     *          be how long we block on QueueTask() being called again.
     *          Returning 0 = Don't care when we get called back. 
     */
    unsigned int RunOnce();

    /*************************************************************************/
private:

    /*************************************************************************/
    /*
     * Initialize DCGM's connection to LOP. 
     * 
     * Returns: DCGM_ST_OK on success.
     *          Other DCGM_ST_? #define on error
     */
    dcgmReturn_t InitLop(void);

    /*************************************************************************/
    /*
     * Helper method to: 
     * 1. Colwert the given group Id to an internal group Id
     * 2. Validate that the group is homogeneous
     * 3. Get the gpuIds of the members of the group
     */
    dcgmReturn_t ValidateGroupIdAndGetGpuIds(dcgmGpuGrp_t grpId, 
                                             unsigned int *groupId,
                                             std::vector<unsigned int> &gpuIds);

    /*************************************************************************/
    /*
     * Build the list of metrics that LOP supports for each GPU.
     */
    void BuildLopMetricList(unsigned int gpuId);

    /*************************************************************************/
    /*
     * Helper method to clean up any LOP watches for a gpu
     */
    void ClearLopWatchesForGpu(dcgm_module_prof_gpu_t *gpu);

    /*************************************************************************/
    /*
     * Sample metrics for a given GPU
     * 
     * gpuWakeupMs OUT: How long until we should revisit this GPU for sampling
     *                  in ms. DCGM_INT32_BLANK = don't care
     * now      IN/OUT: Current timestamp. Update this after each LOP call.
     */
    void ReadGpuMetrics(dcgm_module_prof_gpu_t *gpu, unsigned int *gpuWakeupMs, timelib64_t *now);

    /*************************************************************************/
    /*
     * Print out stats for this module
     * 
     */
    void LogStats(bool logToStdout, bool logToDebugLog);

    /*************************************************************************/
    /*
     * Unwatch all metrics from LOP
     */
    void UnwatchAll(void);

    /*************************************************************************/
    /*
     * Update the clock rates and theoretical bandwidth limits of all GPUs
     */
    void UpdateMemoryClocksAndBw(void);

    /*************************************************************************/
    /*
     * Set elwironmental variables that LOP needs to see
     */
    void SetElwironmentalVariables(void);

    /*************************************************************************/
    /*
     * Read any elwironmental variables that affect this module
     */
    void ReadElwironmentalVariables(void);

    /*************************************************************************/
    /*
     * Helper method to try to add a fieldId and its underlying metric to a GPU
     */
    void TryAddingFieldIdWithMetric(dcgm_module_prof_gpu_t *gpu,
                                    unsigned int fieldId,
                                    dcgmProfMetricIndex_t metricIndex);

    /*************************************************************************/
    void SendGpuMetricsToCache(dcgm_module_prof_gpu_t *gpu, 
                               unsigned int *gpuWakeupMs,
                               timelib64_t *now, DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Helper method to try to add a list of fieldIds as a metric group
     * 
     * Returns: True if any fieldIds in fieldIds were added to a new metric group in mg. 
     *          False if not.
     */
    bool HelperAddFieldsToMetricGroup(dcgm_module_prof_gpu_t *gpu, 
                                      dcgmProfGetMetricGroups_t *mg, 
                                      unsigned short majorId, unsigned short minorId,
                                      std::vector<unsigned short> &fieldIds);

    /*************************************************************************/
    /*
     * Build and cache the supported metric groups for each chip based on the
     * supported fieldIds for this GPU
     */
    void BuildMetricGroupsGV100(dcgm_module_prof_gpu_t *gpu);
    void BuildMetricGroupsTU10x(dcgm_module_prof_gpu_t *gpu);

    /*************************************************************************/
    /* Helpers to cache GPU fields */
    void CacheAvgDoubleValue(dcgm_module_prof_gpu_t *gpu, 
                             unsigned short fieldId,
                             dcgmProfMetricIndex_t metricIndex, 
                             timelib64_t *now,
                             DcgmFvBuffer *fvBuffer);
    void CacheAvgInt64BandwidthValue(dcgm_module_prof_gpu_t *gpu, 
                                     unsigned short fieldId,
                                     dcgmProfMetricIndex_t metricIndex, 
                                     timelib64_t *now,
                                     DcgmFvBuffer *fvBuffer);

    /*************************************************************************/
    /*
     * Helper method to colwert an array of fieldIds to an array of metricIds
     * 
     * An error will be returned if any of the fieldIds aren't supported
     * for the given GPU
     * 
     */
    dcgmReturn_t ColwertFieldIdsToMetricIds(dcgm_module_prof_gpu_t *gpu,
                                            unsigned short *fieldIds, int numFieldIds,
                                            std::vector<dcgmProfMetricIndex_t> &metricIds);

    /*************************************************************************/
    /* 
     * Helper method to colwert a field ID list into an array of metric tags and metric IDs
     * to pass to DcgmLopGpu->InitializeWithMetrics() *
     */
    dcgmReturn_t BuildMetricListFromFieldIdList(dcgm_module_prof_gpu_t *gpu,
                                                std::vector<const char *> (&metricTags)[DLG_MAX_METRIC_GROUPS],
                                                std::vector<unsigned int> (&metricIds)[DLG_MAX_METRIC_GROUPS],
                                                dcgm_profiling_msg_watch_fields_t *msg, 
                                                int *numPassGroups);


    /*************************************************************************/
    /* Helper method of ProcessWatchFields to process a single GPU */
    dcgmReturn_t ProcessWatchFieldsGpu(dcgm_profiling_msg_watch_fields_t *msg,
                                       dcgm_module_prof_gpu_t *gpu);

    /*************************************************************************/
    /* Helper method to read and record all metrics. This is called every
       watch interval from RunOnce() or after watches are first set 
       
       Returns: when this API should be called again to read metrics in ms (from now)
                This is callwlated based on the watch frequency and current timestamp
    */
    unsigned int ReadAndCacheMetrics(void);

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessGetMetricGroups(dcgm_profiling_msg_get_mgs_t *msg);
    dcgmReturn_t ProcessWatchFields(dcgm_profiling_msg_watch_fields_t *msg);
    dcgmReturn_t ProcessUnwatchFields(dcgm_profiling_msg_unwatch_fields_t *msg);
    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg);
    dcgmReturn_t ProcessPauseResume(dcgm_profiling_msg_pause_resume_t *msg);

    /*************************************************************************/
    /* Private member variables */
    DcgmCacheManager *mpCacheManager;   /* Cached pointer to the cache manager. Not owned by this class */
    LwcmGroupManager *mpGroupManager;   /* Cached pointer to the group manager. Not owned by this class */

    std::vector<dcgm_module_prof_gpu_t> m_gpus; /* Represents all of the GPUs in the system. Indexed by gpuId */

    timelib64_t m_cacheIntervalUsec; /* How often we should snapshot values to the cache manager. This is 
                                        lwrrently global */

    bool m_bypassSkuCheck; /* Should we ignore the SKU we are running on? Otherwise, DCGM-DCP is only
                              allowed to run on Tesla V100 SKUs */
    bool m_profilingIsPaused; /* Is profiling lwrrently paused (true) or not (false). When profiling is paused,
                                 blank values should be written to the cache manager */

    /* Stats for this module. Useful for debugging */
    dcgm_module_prof_stats_t m_stats;

    DcgmFvBuffer *m_taskRunnerFvBuffer; /* Buffer for sending field values to the cache manager. This is owned
                                           by the task runner thread. */

    dcgm_connection_id_t m_watcherConnId; /* Connection ID of the client who owns the watches. Note that there 
                                             could still be active watches even if this is 0 if a user decided
                                             to pass persisteAfterDisconnect to dcgmConnect_v2 or is embedded */
};


