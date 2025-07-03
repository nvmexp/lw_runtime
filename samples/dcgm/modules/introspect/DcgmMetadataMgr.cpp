#include "DcgmMetadataMgr.h"
#include "DcgmStringColwersions.h"
#include "LwcmCacheManager.h"
#include "logging.h"
#include "dcgm_fields.h"
#include "timeseries.h"
#include "DcgmMutex.h"
#include "LwcmHostEngineHandler.h"

#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <algorithm>

DcgmMetadataManager::ContextKey::ContextKey()
    : context(STAT_CONTEXT_ILWALID),
      contextId(0),
      aggregate(false),
      fieldScope(-1),
      gpuId(0) {}

DcgmMetadataManager::ContextKey::ContextKey(StatContext context)
    : context(context),
      contextId(0),
      aggregate(false),
      fieldScope(-1),
      gpuId(0)
{}

DcgmMetadataManager::ContextKey::ContextKey(StatContext context,
                                              bool aggregate,
                                              int fieldScope,
                                              unsigned int gpuId)
    : context(context),
      contextId(0),
      aggregate(aggregate),
      fieldScope(fieldScope),
      gpuId(gpuId)
{}

DcgmMetadataManager::ContextKey::ContextKey(StatContext context,
                                              unsigned long long contextId,
                                              bool aggregate,
                                              int fieldScope,
                                              unsigned int gpuId)
    : context(context),
      contextId(contextId),
      aggregate(aggregate),
      fieldScope(fieldScope),
      gpuId(gpuId)
{}

bool DcgmMetadataManager::ContextKey::operator==(const ContextKey &other) const
{
    return context == other.context
        && contextId == other.contextId
        && aggregate == other.aggregate
        && fieldScope == other.fieldScope
        && gpuId == other.gpuId;
};

bool DcgmMetadataManager::ContextKey::isGlobalStat() const
{
    return fieldScope == DCGM_FS_GLOBAL
        || aggregate == true
        || context == STAT_CONTEXT_PROCESS;
};

bool DcgmMetadataManager::ContextKey::isGpuStat() const
{
    return fieldScope == DCGM_FS_DEVICE;
};

string DcgmMetadataManager::ContextKey::str() const
{
    std::stringstream ss;
    ss << "context: " << context << ", contextId: " << contextId
       << " aggregate: " << aggregate << " fieldScope: " << fieldScope
       << " gpuId: " << gpuId;
    return ss.str();
};

const std::string DcgmMetadataManager::STAT_CONTEXT_STRINGS[STAT_CONTEXT_COUNT] = {
    "invalid",
    "field",
    "field-group",
    "all-fields",
    "process",
};

const std::string DcgmMetadataManager::FIELD_METADATA_TYPE_STRINGS[FIELD_MT_COUNT] = {
    "total-bytes-used",
    "total-exec-time",
    "total-fetch-count",
    "mean-update-freq-usec",
    "recent-exec-time",
    "aggregate-instance-count",
};

const std::string DcgmMetadataManager::PROCESS_METADATA_TYPE_STRINGS[PROCESS_MT_COUNT] = {
    "VM_RSS_KB",
    "VM_SWAP_KB",
    "REAL_RAM_KB",
    "PROCESS_TICKS_UTIME",
    "PROCESS_TICKS_STIME",
    "DEVICE_TICKS_TOTAL",
    "CPU_UTIL_UTIME",
    "CPU_UTIL_STIME",
};

template<typename T>
void deleteNotNull(T* &obj)
{
    if (obj)
    {
        delete obj;
        obj = NULL;
    }
}

DcgmMetadataManager::DcgmMetadataManager(DcgmCacheManager* cm) :
        m_cacheManager(cm) {

    m_runIntervalMs = DEFAULT_RUN_INTERVAL_MS;
    m_statCollection = new DcgmStatCollection(false);

    lwosCondCreate(&this->m_metadataUpdatedCondition);
    lwosCondCreate(&this->m_startUpdateCondition);
    m_mutex = new DcgmMutex(0);

    m_lwrrentlyUpdating = false;
    m_updateLoopId = 1;  // not necessary, but easier to read logs if it starts at 1

    m_startOfLwrUpdateLoop = timelib_usecSince1970();

    m_processTypeToOldestKeepTimestamp[PROCESS_MT_TICKS_UTIME] = CPU_AVG_INTERVAL_USEC;
    m_processTypeToOldestKeepTimestamp[PROCESS_MT_TICKS_STIME] = CPU_AVG_INTERVAL_USEC;
    m_processTypeToOldestKeepTimestamp[PROCESS_MT_DEVICE_TICKS_TOTAL] = CPU_AVG_INTERVAL_USEC;

    m_processTypeToMaxKeepEntries[PROCESS_MT_TICKS_UTIME] = 0;
    m_processTypeToMaxKeepEntries[PROCESS_MT_TICKS_STIME] = 0;
    m_processTypeToMaxKeepEntries[PROCESS_MT_DEVICE_TICKS_TOTAL] = 0;

    m_aggregationFunctors.push_back(new AggregateSumFunctor<long long>(this, FIELD_MT_LWR_BYTES_USED));
    m_aggregationFunctors.push_back(new AggregateSumFunctor<long long>(this, FIELD_MT_TOTAL_EXEC_TIME_USEC));
    m_aggregationFunctors.push_back(new AggregateSumFunctor<long long>(this, FIELD_MT_TOTAL_FETCH_COUNT));
    m_aggregationFunctors.push_back(new AggregateSumFunctor<long long>(this, FIELD_MT_AGGR_INSTANCE_COUNT));
    m_aggregationFunctors.push_back(new AggregateMeanFunctor<long long>(this, FIELD_MT_MEAN_UPDATE_FREQ_USEC));

    // this aggregator must come after the aggregator for FIELD_MT_MEAN_UPDATE_FREQ_USEC
    m_aggregationFunctors.push_back(new AggregateNormalizedSumFunctor<double, long long>(this,
                                                                                         FIELD_MT_RECENT_EXEC_TIME_USEC,
                                                                                         FIELD_MT_MEAN_UPDATE_FREQ_USEC));
}

DcgmMetadataManager::~DcgmMetadataManager() {
    int st = StopAndWait(1000);
    if(st)
    {
        PRINT_WARNING("", "Killing introspection thread that is still running.");
        Kill();
    }

    deleteNotNull(m_statCollection);

    lwosCondDestroy(&this->m_metadataUpdatedCondition);
    lwosCondDestroy(&this->m_startUpdateCondition);
    delete(m_mutex);

    for (size_t i = 0; i < m_aggregationFunctors.size(); ++i)
    {
        delete m_aggregationFunctors.at(i);
    }
}

void DcgmMetadataManager::run(void) {

    PRINT_DEBUG("", "MetadataManager started running");

    while (!ShouldStop()) {
        m_startOfLwrUpdateLoop = timelib_usecSince1970();
        PRINT_DEBUG("%d", "starting update loop %d", m_updateLoopId);

        // retrieve "field" metadata and store it at all the aggregated levels that we care about

        // retrieve data for field instances (Global, GPU0, GPU1, ...)
        // and then generate more "field instance"-level data from it (postprocess)
        retrieveFieldInstanceData();
        postProcessFieldInstanceData();

        aggregateFieldData();
        postProcessFieldData();

        aggregateFieldGroupInstanceData();
        postProcessFieldGroupInstanceData();

        aggregateFieldGroupData();
        postProcessFieldGroupData();

        aggregateAllFieldsInstanceData();
        postProcessAllFieldsInstanceData();

        aggregateAllFieldsData();
        postProcessAllFieldsData();

        retrieveProcessData();
        postProcessProcessData();

        PRINT_DEBUG("%d", "done update loop %d", m_updateLoopId);

        dcgm_mutex_lock(m_mutex);

        // wake up external callers that are waiting to read metadata
        lwosCondBroadcast(&m_metadataUpdatedCondition);
        m_lwrrentlyUpdating = false;
        m_updateLoopId++;

        // wait for the next time to update or for an update to be triggered
        m_mutex->CondWait(&m_startUpdateCondition, m_runIntervalMs);

        m_lwrrentlyUpdating = true;

        dcgm_mutex_unlock(m_mutex);
    }

    PRINT_DEBUG("", "Introspection Manager finished running");
}

void DcgmMetadataManager::retrieveFieldInstanceData()
{
    dcgmReturn_t st;

    RetrieveFieldBytesUsedFunctor bytesUsedFunctor(m_cacheManager, this);
    RetrieveFieldTotalExecTimeFunctor totalExecTimeFunctor(m_cacheManager, this);
    RetrieveFieldFetchCountFunctor fetchCountFunctor(m_cacheManager, this);
    RetrieveFieldUpdateFreqFunctor updateFreqFunctor(m_cacheManager, this);
    RetrieveFieldInstanceCountFunctor fieldInstanceCountFunctor(this);

    std::vector<FieldFunctor *> functors;
    functors.push_back(&bytesUsedFunctor);
    functors.push_back(&totalExecTimeFunctor);
    functors.push_back(&fetchCountFunctor);
    functors.push_back(&updateFreqFunctor);
    functors.push_back(&fieldInstanceCountFunctor);

    st = m_cacheManager->ForEachWatchedGlobalField(functors);
    if (st != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "got error %d when retrieving field instance data for global fields", st);
    }

    st = m_cacheManager->ForEachWatchedGpuField(functors);
    if (st != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "got error %d when retrieving field instance data for gpu fields", st);
    }

    PRINT_DEBUG("", "done retrieving field instance data");
}

void DcgmMetadataManager::postProcessFieldInstanceData()
{
    dcgmReturn_t st;
    GenerateFieldRecentUpdateTimeFunctor recentUpdateTimeFunctor(m_cacheManager, this);

    st = m_cacheManager->ForEachWatchedGlobalField(&recentUpdateTimeFunctor);
    if (DCGM_ST_OK != st)
    {
        PRINT_ERROR("%d", "got error %d when post processing field instance data for global fields", st);
    }

    st = m_cacheManager->ForEachWatchedGpuField(&recentUpdateTimeFunctor);
    if (DCGM_ST_OK != st)
    {
        PRINT_ERROR("%d", "got error %d when post processing field instance data for gpu fields", st);
    }

    PRINT_DEBUG("", "done post-processing field instance data");
}

void DcgmMetadataManager::aggregateFieldData()
{
    dcgmReturn_t st;

    FieldAggregator fieldAggregator(this, m_cacheManager, m_aggregationFunctors);

    st = m_cacheManager->ForEachWatchedField(&fieldAggregator);

    if (st != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "got error %d when aggregating field data", st);
    }

    PRINT_DEBUG("", "done aggregating field data");
}

void DcgmMetadataManager::postProcessFieldData()
{
    PRINT_DEBUG("", "done post-processing field data");
}

void DcgmMetadataManager::aggregateFieldGroupInstanceData()
{
    aggregateOverFieldGroupInstances(m_aggregationFunctors);

    PRINT_DEBUG("", "done aggregating field group instance data");
}

void DcgmMetadataManager::postProcessFieldGroupInstanceData()
{
    PRINT_DEBUG("", "done post-processing field group instance data");
}

void DcgmMetadataManager::aggregateFieldGroupData()
{
    aggregateOverFieldGroups(m_aggregationFunctors);

    PRINT_DEBUG("", "done aggregating field group data");
}

void DcgmMetadataManager::postProcessFieldGroupData()
{
    PRINT_DEBUG("", "done post-processing field group data");
}

void DcgmMetadataManager::aggregateAllFieldsInstanceData()
{
    dcgmReturn_t retSt;

    retSt = aggregateOverFieldInstances(ContextKey(STAT_CONTEXT_ALL_FIELDS), m_aggregationFunctors);
    if (DCGM_ST_OK != retSt)
        PRINT_ERROR("%d", "got error %d when aggregating all-fields instance data", retSt);

    PRINT_DEBUG("", "done aggregating all-fields instance data");
}

void DcgmMetadataManager::postProcessAllFieldsInstanceData()
{
    PRINT_DEBUG("", "done post-processing all-fields instance data");
}

void DcgmMetadataManager::aggregateAllFieldsData()
{
    dcgmReturn_t retSt;

    retSt = aggregateOverFields(ContextKey(STAT_CONTEXT_ALL_FIELDS), m_aggregationFunctors, NULL);
    if (DCGM_ST_OK != retSt)
            PRINT_ERROR("%d", "got error %d when aggregating all-fields data", retSt);

    PRINT_DEBUG("", "done aggregating all-fields data");
}

void DcgmMetadataManager::postProcessAllFieldsData()
{
    PRINT_DEBUG("", "done post-processing all-fields data");
}

void DcgmMetadataManager::retrieveProcessData()
{
    retrieveOSInfo();
    PRINT_DEBUG("", "done retrieving OS process data");
}

void DcgmMetadataManager::postProcessProcessData()
{
    generateCpuUtilization();
    PRINT_DEBUG("", "done post-processing OS process data");
}

template <typename Fn>
dcgmReturn_t DcgmMetadataManager::forEachInstanceContext(Fn *fn, std::vector<unsigned short> *fieldIds)
{
    FieldInstanceToContextAdaptor<Fn> it(fn);
    dcgmReturn_t st1 = m_cacheManager->ForEachWatchedGpuField(&it, fieldIds);
    dcgmReturn_t st2 = m_cacheManager->ForEachWatchedGlobalField(&it, fieldIds);

    if (DCGM_ST_OK != st1)
        return st1;
    return st2;
}

dcgmReturn_t DcgmMetadataManager::aggregateOverFieldGroups(std::vector<AggregateFunctor *> functors)
{
    dcgmReturn_t st = DCGM_ST_OK;
    dcgmReturn_t retSt = st;
    dcgmAllFieldGroup_t allGroupInfo;
    dcgmFieldGroupInfo_t *fgi = 0;
    int i;

    DcgmFieldGroupManager *fieldGroupManager = LwcmHostEngineHandler::Instance()->GetFieldGroupManager();

    st = fieldGroupManager->PopulateFieldGroupGetAll(&allGroupInfo);
    if(st != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "PopulateFieldGroupGetAll returned %d", (int)st);
        return st;
    }

    for (i = 0; i < (int)allGroupInfo.numFieldGroups; i++)
    {
        fgi = &allGroupInfo.fieldGroups[i];
        std::vector<unsigned short>fieldIdsVec(fgi->fieldIds, fgi->fieldIds + fgi->numFieldIds);

        ContextKey aggrContext(STAT_CONTEXT_FIELD_GROUP);
        aggrContext.contextId = (uintptr_t)fgi->fieldGroupId;

        st = aggregateOverFields(aggrContext, functors, &fieldIdsVec);
        if (DCGM_ST_OK != st)
        {
            retSt = st;
            PRINT_ERROR("%d %llu", "got error %d when aggregating fields in field group %llu",
                        (int)st, (unsigned long long)(uintptr_t)fgi->fieldGroupId);
        }
    }

    return retSt;
}

dcgmReturn_t DcgmMetadataManager::aggregateOverFieldGroupInstances(std::vector<AggregateFunctor *> functors)
{
    dcgmReturn_t st = DCGM_ST_OK;
    dcgmReturn_t retSt = st;
    dcgmAllFieldGroup_t allGroupInfo;
    dcgmFieldGroupInfo_t *fgi = 0;
    int i;

    DcgmFieldGroupManager *fieldGroupManager = LwcmHostEngineHandler::Instance()->GetFieldGroupManager();

    st = fieldGroupManager->PopulateFieldGroupGetAll(&allGroupInfo);
    if(st != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "PopulateFieldGroupGetAll returned %d", (int)st);
        return st;
    }                                    

    for (i = 0; i < (int)allGroupInfo.numFieldGroups; i++)
    {
        fgi = &allGroupInfo.fieldGroups[i];
        std::vector<unsigned short> fieldIdsVec(fgi->fieldIds, fgi->fieldIds + fgi->numFieldIds);

        ContextKey aggrContext(STAT_CONTEXT_FIELD_GROUP);
        aggrContext.contextId = (uintptr_t)fgi->fieldGroupId;

        st = aggregateOverFieldInstances(aggrContext,
                                         functors, &fieldIdsVec);
        if (DCGM_ST_OK != st)
        {
            retSt = st;
            PRINT_ERROR("%d %llu", "got error %d when aggregating field instances in field group %llu",
                        st, (unsigned long long)(uintptr_t)fgi->fieldGroupId);
        }
    }

    return retSt;
}

dcgmReturn_t DcgmMetadataManager::aggregateOverFieldInstances(ContextKey partialAggregateContext,
                                                              std::vector<AggregateFunctor *> functors,
                                                              std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st = DCGM_ST_OK;
    dcgmReturn_t retSt = st;

    if (m_cacheManager->AnyGlobalFieldsWatched(fieldIds))
    {
        // for each aggregator, aggregate over all global field instances
        for (size_t i = 0; i < functors.size(); ++i)
        {
            AggregateFunctor *functor = functors.at(i);
            FieldInstanceToContextAdaptor<AggregateFunctor> wrappedFunctor(functor);

            ContextKey globalAggContext(partialAggregateContext);
            globalAggContext.aggregate = false;
            globalAggContext.fieldScope = DCGM_FS_GLOBAL;

            st = functor->initAggregation(globalAggContext);
            if (DCGM_ST_OK != st)
            {
                retSt = st;
                continue;
            }

            m_cacheManager->ForEachWatchedGlobalField(&wrappedFunctor, fieldIds);
            if (DCGM_ST_OK != st)
            {
                retSt = st;
                continue;
            }

            functor->recordIfOkay();
            if (DCGM_ST_OK != st)
            {
                retSt = st;
                continue;
            }
        }

        if (DCGM_ST_OK != retSt)
        {
            PRINT_ERROR("%d", "got error %d when aggregating all-fields instance data for global field instances",
                        retSt);
        }
    }


    retSt = DCGM_ST_OK;

    std::vector<unsigned int> gpuIds;
    st = m_cacheManager->GetGpuIds(1, gpuIds);
    if (DCGM_ST_OK != st)
    {
        PRINT_ERROR("", "failed to get gpuIds");
    }
    else
    {
        // for each functor, aggregate all field instances for 1 gpu at a time
        for (size_t i = 0; i < functors.size(); ++i)
        {
            AggregateFunctor *functor = functors.at(i);
            FieldInstanceToContextAdaptor<AggregateFunctor> wrappedFunctor(functor);

            for (size_t ii = 0; ii < gpuIds.size(); ++ii)
            {
                unsigned gpuId = gpuIds.at(ii);
                bool isWatched;

                if (!m_cacheManager->AnyGpuFieldsWatched(gpuId, fieldIds))
                {
                    break;
                }


                ContextKey gpuAggContext(partialAggregateContext);
                gpuAggContext.aggregate = false;
                gpuAggContext.fieldScope = DCGM_FS_DEVICE;
                gpuAggContext.gpuId = gpuId;

                st = functor->initAggregation(gpuAggContext);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }

                m_cacheManager->ForEachWatchedGpuField(&wrappedFunctor, fieldIds, &gpuId);
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }

                functor->recordIfOkay();
                if (DCGM_ST_OK != st)
                {
                    retSt = st;
                    continue;
                }
            }

            if (DCGM_ST_OK != retSt)
            {
                PRINT_ERROR("%d %zu", "got error %d when aggregating all-fields instance data for gpu field instances on aggregator %zu",
                            retSt, i);
            }
        }
    }

    return retSt;
}


dcgmReturn_t DcgmMetadataManager::aggregateOverFields(ContextKey partialAggregateContext,
                                                      std::vector<AggregateFunctor *> functors,
                                                      std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st = DCGM_ST_OK;
    dcgmReturn_t retSt = st;

    // for each aggregator, aggregate over all field instances.
    for (size_t i = 0; i < functors.size(); ++i)
    {
        AggregateFunctor *functor = functors.at(i);

        ContextKey aggrContext(partialAggregateContext);
        aggrContext.aggregate = true;

        st = functor->initAggregation(aggrContext);
        if (DCGM_ST_OK != st && DCGM_ST_NO_DATA != st && DCGM_ST_NOT_WATCHED != st)
        {
            retSt = st;
            goto PRINT_AGGREGATE_ALL_FIELDS_ERROR;
        }

        st = forEachInstanceContext(functor, fieldIds);
        if (DCGM_ST_OK != st && DCGM_ST_NO_DATA != st && DCGM_ST_NOT_WATCHED != st)
        {
            retSt = st;
            goto PRINT_AGGREGATE_ALL_FIELDS_ERROR;
        }

        functor->recordIfOkay();
        if (DCGM_ST_OK != st && DCGM_ST_NO_DATA != st && DCGM_ST_NOT_WATCHED != st)
        {
            retSt = st;
            goto PRINT_AGGREGATE_ALL_FIELDS_ERROR;
        }

PRINT_AGGREGATE_ALL_FIELDS_ERROR:
        if (DCGM_ST_OK != retSt)
        {
            PRINT_ERROR("%d %zu", "got error %d when aggregating all-fields on aggregator %zu",
                        retSt, i);
        }
    }

    return retSt;
}

dcgmReturn_t DcgmMetadataManager::lastUpdateTime(StatKey sKey, timelib64_t *updateTimeUsec)
{
    mcollect_value_p measurement = getStatMeasurement(sKey);

    if (measurement == NULL)
    {
        PRINT_DEBUG("%s", "measurement for stat \"%s\" has no records", sKey.str().c_str());
        return DCGM_ST_NO_DATA;
    }
    else if (measurement->type != MC_TYPE_TIMESERIES_DOUBLE
          && measurement->type != MC_TYPE_TIMESERIES_INT64
          && measurement->type != MC_TYPE_TIMESERIES_STRING
          && measurement->type != MC_TYPE_TIMESERIES_BLOB)
    {
        PRINT_ERROR("", "measurement is not a timeseries");
        return DCGM_ST_BADPARAM;
    }

    int measurementCount = timeseries_size(measurement->val.tseries);
    if (measurementCount == 0)
    {
        PRINT_DEBUG("%s", "measurement for stat \"%s\" has no records", sKey.str().c_str());
        return DCGM_ST_NO_DATA;
    }

    *updateTimeUsec = timeseries_last(measurement->val.tseries, NULL)->usecSince1970;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMetadataManager::generateFieldRecentUpdateTime(unsigned short fieldId, unsigned int gpuId)
{
    dcgmReturn_t st;
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

    StatKey totalExecTimeKey(ContextKey(STAT_CONTEXT_FIELD, fieldId, false, fieldMeta->scope, gpuId),
                             FIELD_MT_TOTAL_EXEC_TIME_USEC);

    mcollect_value_p measurement = getStatMeasurement(totalExecTimeKey);

    st = validateMcollectType(measurement, FIELD_MT_TOTAL_EXEC_TIME_USEC);
    if (DCGM_ST_OK != st)
        return st;

    long long recentExecTime;
    int measurementCount = timeseries_size(measurement->val.tseries);
    if (measurementCount == 0)
    {
        return DCGM_ST_OK;  // field never retrieved, nothing to do
    }
    else if (measurementCount == 1)
    {
        recentExecTime = timeseries_last(measurement->val.tseries, NULL)->val.i64;
    }
    else
    {
        st = getRecentStatDiff(measurement->val.tseries, &recentExecTime);
        if (DCGM_ST_OK != st)
        {
            return st;
        }
        if (recentExecTime == 0)
        {
            return DCGM_ST_OK; // when last exelwtion time was retrieved, no update had been done
        }
    }

    // store recent exec time as a double because it will get normalized when it is aggregated later
    // and we do not want to lose info from truncation
    StatKey recentExecTimeKey(ContextKey(STAT_CONTEXT_FIELD, fieldId, false, fieldMeta->scope, gpuId),
                             FIELD_MT_RECENT_EXEC_TIME_USEC);
    recordStat(recentExecTimeKey, (double)recentExecTime);

    return DCGM_ST_OK;
}

timelib64_t DcgmMetadataManager::getOldestKeepTimestamp(ProcessMetadataType mType)
{
    if (m_processTypeToOldestKeepTimestamp.count(mType) == 1)
    {
        return m_processTypeToOldestKeepTimestamp[mType];
    }
    else
    {
        return DEFAULT_OLDEST_KEEP_TIMESTAMP;
    }
}

timelib64_t DcgmMetadataManager::getOldestKeepTimestamp(FieldMetadataType mType)
{
    if (m_fieldTypeToOldestKeepTimestamp.count(mType) == 1)
    {
        return m_fieldTypeToOldestKeepTimestamp[mType];
    }
    else
    {
        return DEFAULT_OLDEST_KEEP_TIMESTAMP;
    }
}

int DcgmMetadataManager::getMaxKeepEntries(ProcessMetadataType mType)
{
    if (m_processTypeToMaxKeepEntries.count(mType) == 1)
    {
        return m_processTypeToMaxKeepEntries[mType];
    }
    else
    {
        return DEFAULT_MAX_KEEP_ENTRIES;
    }
}

int DcgmMetadataManager::getMaxKeepEntries(FieldMetadataType mType)
{
    if (m_fieldTypeToMaxKeepEntries.count(mType) == 1)
    {
        return m_fieldTypeToMaxKeepEntries[mType];
    }
    else
    {
        return DEFAULT_MAX_KEEP_ENTRIES;
    }
}

dcgmReturn_t DcgmMetadataManager::getRecentStatDiff(timeseries_p ts, long long *diff)
{
    timeseries_lwrsor_t cursor;
    timeseries_entry_p lastStatEntry = timeseries_last(ts, &cursor);
    if (lastStatEntry == NULL)
    {
        return DCGM_ST_NO_DATA;
    }

    timeseries_entry_p prevStatEntry = timeseries_prev(ts, &cursor);
    if (prevStatEntry == NULL)
    {
        return DCGM_ST_NO_DATA;
    }

    *diff = lastStatEntry->val.i64 - prevStatEntry->val.i64;

    return DCGM_ST_OK;
}

template <typename StatT, typename NormT, typename RetT>
dcgmReturn_t DcgmMetadataManager::getNormalizedLatestStat(const StatKey &statKey,
                                                          const StatKey &normFromStatKey,
                                                          const StatKey &normToStatKey,
                                                          RetT *normalizedStat)
{
    dcgmReturn_t st;
    StatT stat;
    NormT normFrom;
    NormT normTo;

    st = getStat(statKey, &stat);
    if (DCGM_ST_OK != st)
        return st;

    st = getStat(normFromStatKey, &normFrom);
    if (DCGM_ST_OK != st)
        return st;

    st = getStat(normToStatKey, &normTo);
    if (DCGM_ST_OK != st)
        return st;

    if (normTo == 0)
    {
        PRINT_ERROR("%s", "cannot normalize from 0. key: %s", normFromStatKey.str().c_str());
        return DCGM_ST_BADPARAM;
    }

    return static_cast<RetT>(stat * ((double)normTo / (double)(normFrom)));
}

string DcgmMetadataManager::StatKey::str() const
{
    string aggregateStr = cKey.aggregate ? "aggregate" : "";
    string contextStr = STAT_CONTEXT_STRINGS[cKey.context];
    string contextIdStr;
    string mTypeStr;

    if (cKey.context == STAT_CONTEXT_FIELD)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(cKey.contextId);
        if (fieldMeta && fieldMeta->fieldId != DCGM_FI_UNKNOWN)
        {
            contextIdStr = string(fieldMeta->tag);
        }
        else
        {
            contextIdStr = toStr(cKey.contextId);
        }
    }
    else if (cKey.context == STAT_CONTEXT_FIELD_GROUP)
    {
        DcgmFieldGroupManager *fieldGroupManager = LwcmHostEngineHandler::Instance()->GetFieldGroupManager();

        contextIdStr = fieldGroupManager->GetFieldGroupName((dcgmFieldGrp_t)cKey.contextId);
        if (contextIdStr.compare("") == 0)
        {
            contextIdStr = toStr(cKey.contextId);
        }
    }
    else
    {
        contextIdStr = "";
    }

    if (cKey.context == STAT_CONTEXT_PROCESS)
    {
        mTypeStr = PROCESS_METADATA_TYPE_STRINGS[mType.process];
    }
    else
    {
        mTypeStr = FIELD_METADATA_TYPE_STRINGS[mType.field];
    }

    return aggregateStr + ":" + contextStr + ":" + contextIdStr + ":" + mTypeStr;
}

template <typename T>
dcgmReturn_t DcgmMetadataManager::getRecentStat(StatKey sKey, T *metadata)
{
    timelib64_t statUpdateTime;
    dcgmReturn_t st = lastUpdateTime(sKey, &statUpdateTime);
    if (DCGM_ST_OK != st)
        return st;

    if (statUpdateTime < m_startOfLwrUpdateLoop)
    {
        PRINT_DEBUG("%s", "no stat was found to be recorded for key %s since the start of the last (or current) update loop. "
                    " double check when this stat is recorded.", sKey.str().c_str());
        return DCGM_ST_STALE_DATA;
    }

    return getStat(sKey, metadata);
}

template <typename T>
dcgmReturn_t DcgmMetadataManager::getStat(StatKey sKey, T *metadata)
{
    dcgmReturn_t retSt;

    if (metadata == NULL)
    {
        PRINT_ERROR("", "param cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    string statKey = sKey.str();

    {
        // hold lock while in this block
        DcgmLockGuard lock(m_mutex);
        mcollect_value_p measurement = getStatMeasurement(sKey);

        if (sKey.cKey.context == STAT_CONTEXT_PROCESS)
        {
            retSt = validateMcollectType(measurement, sKey.mType.process);
        }
        else
        {
            retSt = validateMcollectType(measurement, sKey.mType.field);
        }

        if (retSt != DCGM_ST_OK)
        {
            PRINT_DEBUG("%s", "no metadata found for key %s", statKey.c_str());
            return retSt;
        }

        timeseries_p ts = measurement->val.tseries;
        timeseries_lwrsor_t tsLwrsor;
        timeseries_entry_p tsVal = timeseries_last(ts, &tsLwrsor);
        if (tsVal == NULL)
        {
            PRINT_DEBUG("%s", "no metadata found for key %s", statKey.c_str());
            return DCGM_ST_NO_DATA;
        }

        if (sKey.cKey.context == STAT_CONTEXT_PROCESS)
        {
            extractTimeseriesVal(metadata, tsVal, sKey.mType.process);
        }
        else
        {
            extractTimeseriesVal(metadata, tsVal, sKey.mType.field);
        }
    }

    return DCGM_ST_OK;
}

template <typename T>
dcgmReturn_t DcgmMetadataManager::recordStat(StatKey sKey, const T &val)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    timelib64_t oldestKeepTimestamp;
    int maxKeepEntries;
    string statKey = sKey.str();

    if (sKey.cKey.context == STAT_CONTEXT_PROCESS)
    {
        oldestKeepTimestamp = getOldestKeepTimestamp(sKey.mType.process);
        maxKeepEntries = getMaxKeepEntries(sKey.mType.process);
    }
    else {
        oldestKeepTimestamp = getOldestKeepTimestamp(sKey.mType.field);
        maxKeepEntries = getMaxKeepEntries(sKey.mType.field);
    }

    timelib64_t now = timelib_usecSince1970();

    dcgm_mutex_lock(m_mutex);

    if (sKey.cKey.isGpuStat())
    {
        m_statCollection->AppendEntityStat(SC_ENTITY_GROUP_GPU, m_cacheManager->GpuIdToLwmlIndex(sKey.cKey.gpuId), statKey, val, 0, now);
        m_statCollection->EnforceEntityStatQuota(SC_ENTITY_GROUP_GPU, m_cacheManager->GpuIdToLwmlIndex(sKey.cKey.gpuId),
                                                 statKey, oldestKeepTimestamp, maxKeepEntries);
        PRINT_DEBUG("%u %s %f %lld", "Recorded metadata stat gpuId %u, key %s, val %f, now %lld", 
                    sKey.cKey.gpuId, statKey.c_str(), (double)val, (long long)now);
    }
    else if (sKey.cKey.isGlobalStat())
    {
        m_statCollection->AppendGlobalStat(statKey, val, now);
        m_statCollection->EnforceGlobalStatQuota(statKey, oldestKeepTimestamp, maxKeepEntries);
        PRINT_DEBUG("%s %f %lld", "Recorded metadata stat (global), key %s, val %f, now %lld",
                    statKey.c_str(), (double)val, (long long)now);
    }
    else
    {
        PRINT_ERROR("%s", "invalid stat context, %s", sKey.cKey.str().c_str());
        retSt = DCGM_ST_BADPARAM;
    }

    dcgm_mutex_unlock(m_mutex);

    return retSt;
}

template <typename MetadataType>
void DcgmMetadataManager::extractTimeseriesVal(void *val, timeseries_entry_p tsVal, MetadataType mType)
{
    int mcType = mcollectTypeForMetadataType(mType);
    if (mcType == MC_TYPE_TIMESERIES_INT64)
    {
        *(long long *)val = tsVal->val.i64;
    }
    else if (mcType == MC_TYPE_TIMESERIES_DOUBLE)
    {
        *(double *)val = tsVal->val.dbl;
    }
    else if (mcType == MC_TYPE_TIMESERIES_BLOB)
    {
        memcpy(val, tsVal->val.ptr, (size_t)tsVal->val2.ptrSize);
    }
    else
    {
        PRINT_ERROR("%d %d", "unknown mcType \"%d\" from metadata type %d, see MC_TYPE_? for values",
                    mcType, mType);
    }
}

dcgmReturn_t DcgmMetadataManager::SetRunInterval(unsigned int intervalMs)
{
    dcgm_mutex_lock(m_mutex);
    m_runIntervalMs = intervalMs;
    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

template <typename MetadataType>
dcgmReturn_t DcgmMetadataManager::validateMcollectType(mcollect_value_p measurement, MetadataType mType)
{
    if (NULL == measurement)
    {
        return DCGM_ST_NO_DATA;
    }

    int expectedType = mcollectTypeForMetadataType(mType);
    if (measurement->type != expectedType)
    {
        PRINT_ERROR("%d %d %d", "INTERNAL ERROR: measurement with MetadataType \"%d\"did not have expected type of %d. "
                    "Got %d.  See MC_TYPE_? for types.",
                    (int)mType, expectedType, measurement->type);
        return DCGM_ST_GENERIC_ERROR;
    }
    return DCGM_ST_OK;
}

int DcgmMetadataManager::mcollectTypeForMetadataType(FieldMetadataType mType) const
{
    switch (mType)
    {
        case FIELD_MT_LWR_BYTES_USED:
            return MC_TYPE_TIMESERIES_INT64;
        case FIELD_MT_TOTAL_EXEC_TIME_USEC:
            return MC_TYPE_TIMESERIES_INT64;
        case FIELD_MT_TOTAL_FETCH_COUNT:
            return MC_TYPE_TIMESERIES_INT64;
        case FIELD_MT_MEAN_UPDATE_FREQ_USEC:
            return MC_TYPE_TIMESERIES_INT64;
        case FIELD_MT_RECENT_EXEC_TIME_USEC:
            return MC_TYPE_TIMESERIES_DOUBLE;
        case FIELD_MT_AGGR_INSTANCE_COUNT:
            return MC_TYPE_TIMESERIES_INT64;
        default:
            PRINT_ERROR("%d", "invalid metadata type of %d", mType);
            return MC_TYPE_UNKNOWN;
    }
}

int DcgmMetadataManager::mcollectTypeForMetadataType(ProcessMetadataType mType) const
{
    switch (mType)
    {
        case PROCESS_MT_VM_RSS_KB:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_VM_SWAP_KB:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_REAL_RAM_KB:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_TICKS_UTIME:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_TICKS_STIME:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_DEVICE_TICKS_TOTAL:
            return MC_TYPE_TIMESERIES_INT64;
        case PROCESS_MT_CPU_UTIL_UTIME:
            return MC_TYPE_TIMESERIES_DOUBLE;
        case PROCESS_MT_CPU_UTIL_STIME:
            return MC_TYPE_TIMESERIES_DOUBLE;
        default:
            PRINT_ERROR("%d", "invalid metadata type of %d", mType);
            return MC_TYPE_UNKNOWN;
    }
}

template <typename Fn>
dcgmReturn_t DcgmMetadataManager::getMetadataWithWait(Fn getMetadataFn, bool waitIfNoData)
{
    dcgmReturn_t status = DCGM_ST_OK;

    for (int attempt = 1; attempt <= 5; ++attempt)
    {
        // only get metadata when an update loop is not olwrring. There is
        // still a chance for it to start after this line which means there is a small
        // chance for a dirty read but this is less important than grabbing a lock for
        // retrieval and potentially starving the update loop
        waitForUpdateLoopToFinish();
        status = getMetadataFn();
        if (!waitIfNoData || status != DCGM_ST_NO_DATA)
            break;

        // re-attempt retrieval if we are supposed to wait when there is no metadata
        // this could end up waiting when it doesn't have to since no lock is used
        // at this level but it could only happen once at startup
        else
        {
            PRINT_DEBUG("%d", "no metadata available; waiting after retrieval attempt %d", attempt);

            dcgm_mutex_lock(m_mutex);

            // avoid spurious wakeups
            int startULID = m_updateLoopId;
            while (m_updateLoopId == startULID)
            {
                m_mutex->CondWait(&this->m_metadataUpdatedCondition, m_runIntervalMs + 1000);
            }

            dcgm_mutex_unlock(m_mutex);
        }
    }

    if (waitIfNoData && DCGM_ST_OK != status)
    {
        PRINT_WARNING("%d", "INTERNAL ERROR: failed to retrieve data for metadata after it should be populated, error: %d. "
                      "This means that the watch hasn't triggered for this field yet or that the field is unsupported even though it is watched.",
                      status);
    }

    return status;
}

bool DcgmMetadataManager::isValidContext(ContextKey c)
{
    switch (c.context)
    {
    case STAT_CONTEXT_FIELD:
    {
        // validate things specific to a single field
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(c.contextId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            return false;
        }

        // field scope only needs to be correct if the context is
        // not an aggregate of all fields scopes
        if (!c.aggregate && fieldMeta->scope != c.fieldScope)
        {
            return false;
        }

        return isValidMultiFieldContext(c);
    }
    case STAT_CONTEXT_FIELD_GROUP:
    {
        return isValidMultiFieldContext(c);
    }
    case STAT_CONTEXT_ALL_FIELDS:
    {
        return isValidMultiFieldContext(c);
    }
    case STAT_CONTEXT_PROCESS:
    {
        return true; // nothing else about the ContextKey is used if the context is "process"
    }
    default:
    {
        return false;
    }
    }
}

bool DcgmMetadataManager::isValidMultiFieldContext(ContextKey c)
{
    // if it is aggregate, nothing else is used
    if (c.aggregate)
    {
        return true;
    }

    // validation per field scope
    switch (c.fieldScope)
    {
    case DCGM_FS_GLOBAL:
    {
        return true;
    }
    case DCGM_FS_DEVICE:
    {
        std::vector<unsigned int> gpuIds;
        m_cacheManager->GetGpuIds(1, gpuIds);

        // is the GPU ID valid?
        if (std::count(gpuIds.begin(), gpuIds.end(), c.gpuId) >= 1)
        {
            return true;
        }
        else {
            return false;
        }
    }
    default:
    {
        return false;
    }
    }
}

bool DcgmMetadataManager::isContextWatched(ContextKey context)
{
    switch (context.context)
    {
    case STAT_CONTEXT_FIELD:
    {
        dcgmReturn_t st = DCGM_ST_OK;
        bool isWatched = false;
        std::vector<unsigned short> fieldIds;
        fieldIds.push_back(context.contextId);

        if (context.aggregate)
        {
            return this->m_cacheManager->AnyFieldsWatched(&fieldIds);
        }

        if (context.fieldScope == DCGM_FS_GLOBAL)
        {
            st = this->m_cacheManager->IsGlobalFieldWatched(context.contextId, &isWatched);
        }
        else if (context.fieldScope == DCGM_FS_DEVICE)
        {
            st = this->m_cacheManager->IsGpuFieldWatchedOnAnyGpu(context.contextId, &isWatched);
        }

        return (st == DCGM_ST_OK) && isWatched;
    }
    case STAT_CONTEXT_FIELD_GROUP:
    {
        dcgmReturn_t dcgmReturn;
        std::vector<unsigned short> fieldIdsVec;
        DcgmFieldGroupManager *fieldGroupManager = LwcmHostEngineHandler::Instance()->GetFieldGroupManager();

        dcgmReturn = fieldGroupManager->GetFieldGroupFields((dcgmFieldGrp_t)context.contextId, fieldIdsVec);
        if(dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d", "GetFieldGroupFields returned %d", (int)dcgmReturn);
            return false;
        }

        if (context.aggregate)
        {
            return m_cacheManager->AnyFieldsWatched(&fieldIdsVec);
        }

        if (context.fieldScope == DCGM_FS_GLOBAL)
        {
            return m_cacheManager->AnyGlobalFieldsWatched(&fieldIdsVec);
        }
        else if (context.fieldScope == DCGM_FS_DEVICE)
        {
            return m_cacheManager->AnyGpuFieldsWatchedAnywhere(&fieldIdsVec);
        }
        else
        {
            return false; // invalid scope
        }
    }
    case STAT_CONTEXT_ALL_FIELDS:
    {
        if (context.aggregate)
        {
            return m_cacheManager->AnyFieldsWatched(NULL);
        }

        if (context.fieldScope == DCGM_FS_GLOBAL)
        {
            return m_cacheManager->AnyGlobalFieldsWatched(NULL);
        }
        else if (context.fieldScope == DCGM_FS_DEVICE)
        {
            return m_cacheManager->AnyGpuFieldsWatchedAnywhere(NULL);
        }
        else
        {
            return false; // invalid scope
        }
    }
    case STAT_CONTEXT_PROCESS:
    {
        return true;
    }
    default:
    {
        return false; // invalid context
    }
    }
}

dcgmReturn_t DcgmMetadataManager::validateRetrievalContext(ContextKey context)
{
    if (!isValidContext(context))
    {
        PRINT_ERROR("%s", "context is invalid: %s", context.str().c_str());
        return DCGM_ST_BADPARAM;
    }

    if (!isContextWatched(context))
    {
        PRINT_ERROR("%s", "introspection data is not being gathered for context \"%s\"", context.str().c_str());
        return DCGM_ST_NOT_WATCHED;
    }

    return DCGM_ST_OK;
}

void DcgmMetadataManager::waitForUpdateLoopToFinish()
{
    DcgmLockGuard lock(m_mutex);
    while (m_lwrrentlyUpdating)
        m_mutex->CondWait(&this->m_metadataUpdatedCondition, 0);
}

dcgmReturn_t DcgmMetadataManager::GetBytesUsed(ContextKey context, long long *pTotalBytesUsed, bool waitIfNoData)
{
    switch (context.context)
    {
    case STAT_CONTEXT_FIELD:               // fall through
    case STAT_CONTEXT_FIELD_GROUP:         // fall through
    case STAT_CONTEXT_ALL_FIELDS:          // fall through
    case STAT_CONTEXT_PROCESS:
        break;
    default:
        PRINT_ERROR("%d", "invalid StatContext: %d", (int)context.context);
        return DCGM_ST_BADPARAM;
    }

    if (pTotalBytesUsed == NULL)
    {
        PRINT_ERROR("", "arg cannot be null");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t st = validateRetrievalContext(context);
    if (DCGM_ST_OK != st)
        return st;

    if (context.context == STAT_CONTEXT_PROCESS)
    {
        return getHostengineBytesUsed(context, pTotalBytesUsed, waitIfNoData);
    }
    else
    {
        return getFieldStatBytesUsed(context, pTotalBytesUsed, waitIfNoData);
    }
}

dcgmReturn_t DcgmMetadataManager::getHostengineBytesUsed(ContextKey context, long long *bytesUsed, bool waitIfNoData)
{
    StatKey sKey(context, PROCESS_MT_REAL_RAM_KB);
    GetStatFunctor<long long> functor(this, sKey, bytesUsed);
    dcgmReturn_t st = getMetadataWithWait(functor, waitIfNoData);
    if (DCGM_ST_OK == st)
    {
        *bytesUsed = *bytesUsed * 1024;
    }
    return st;
}

dcgmReturn_t DcgmMetadataManager::getFieldStatBytesUsed(ContextKey context, long long *bytesUsed, bool waitIfNoData)
{
    StatKey sKey(context, FIELD_MT_LWR_BYTES_USED);
    GetStatFunctor<long long> functor(this, sKey, bytesUsed);
    return getMetadataWithWait(functor, waitIfNoData);
}

dcgmReturn_t DcgmMetadataManager::GetExecTime(ContextKey context, ExecTimeInfo *execTime, bool waitIfNoData)
{
    switch (context.context)
    {
    case STAT_CONTEXT_FIELD:               // fall through
    case STAT_CONTEXT_FIELD_GROUP:         // fall through
    case STAT_CONTEXT_ALL_FIELDS:
        break;
    default:
        PRINT_ERROR("%d", "invalid StatContext: %d", (int)context.context);
        return DCGM_ST_BADPARAM;
    }

    if (execTime == NULL)
    {
        PRINT_ERROR("", "arg cannot be null");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t st = validateRetrievalContext(context);
    if (DCGM_ST_OK != st)
        return st;

    GetStatFunctor<long long> meanFreqFn(this,
                                         StatKey(context, FIELD_MT_MEAN_UPDATE_FREQ_USEC),
                                         &execTime->meanFrequencyUsec);
    st = getMetadataWithWait(meanFreqFn, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    GetStatFunctor<double> recentUpdateTimeFn(this,
                                              StatKey(context, FIELD_MT_RECENT_EXEC_TIME_USEC),
                                              &execTime->recentUpdateUsec);
    st = getMetadataWithWait(recentUpdateTimeFn, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    GetStatFunctor<long long> totExecTimeFn(this,
                                            StatKey(context, FIELD_MT_TOTAL_EXEC_TIME_USEC),
                                            &execTime->totalEverUpdateUsec);
    st = getMetadataWithWait(totExecTimeFn, waitIfNoData);
    if (DCGM_ST_OK != st)
        return st;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMetadataManager::GetCpuUtilization(CpuUtil *cpuUtil, bool waitIfNoData)
{
    if (cpuUtil == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    GetProcessCpuUtilFunctor functor(this, cpuUtil);
    return getMetadataWithWait(functor, waitIfNoData);
}

dcgmReturn_t DcgmMetadataManager::getCpuUtilizationForHostengine(CpuUtil *cpuUtil)
{
    dcgmReturn_t status;

    StatKey uTimeKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_CPU_UTIL_UTIME);
    StatKey sTimeKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_CPU_UTIL_STIME);

    double cpuUtilUtime;
    double cpuUtilStime;

    status = getStat(uTimeKey, &cpuUtilUtime);
    if (DCGM_ST_OK != status)
        return status;

    status = getStat(sTimeKey, &cpuUtilStime);
    if (DCGM_ST_OK != status)
        return status;

    cpuUtil->kernel = cpuUtilStime;
    cpuUtil->user = cpuUtilUtime;
    cpuUtil->total = cpuUtilStime + cpuUtilUtime;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMetadataManager::UpdateAll(int waitForUpdate)
{
    dcgm_mutex_lock(m_mutex);

    // wait for the current update loop to finish if it is in progress.
    // need a while loop because of "spurious wakeups" (see pthread_cond_wait(3) - Linux man page)
    int startULID = m_updateLoopId;
    while (m_lwrrentlyUpdating || startULID == m_updateLoopId) {
        m_mutex->CondWait(&m_metadataUpdatedCondition, LWOS_INFINITE_TIMEOUT);
    }

    // trigger an update to occur
    lwosCondSignal(&m_startUpdateCondition);

    // wait for the update to finish if we were told to
    if (waitForUpdate) {
        // again, wonderful spurious wakeups force us to loop
        startULID = m_updateLoopId;
        while (startULID == m_updateLoopId)
            m_mutex->CondWait(&m_metadataUpdatedCondition, LWOS_INFINITE_TIMEOUT);
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

void DcgmMetadataManager::retrieveOSInfo()
{
    retrieveOSInfoFromProcSelfStatus();
    retrieveOSInfoFromProcSelfStat();
    retrieveOSInfoFromProcStat();
}

void DcgmMetadataManager::retrieveOSInfoFromProcSelfStatus()
{
    // parse /proc/self/status instead of /proc/self/stat since the swap related
    // fields in "man proc" for /proc/self/stat say (not maintained)
    // /status is also displayed in KB values instead of pages for RSS which is easier to use
    std::ifstream statusStream("/proc/self/status");
    dcgmReturn_t status;

    long long rss = 0;
    long long swp = 0;

    std::string line;
    while (getline(statusStream, line))
    {
        std::stringstream ss(line);
        std::string first, second;

        ss >> first >> second;

        if (first == "VmRSS:") {
            rss = strTo<long long>(second);
        }
        // VmSwap shows up but is no longer dolwmented in "man proc" so we might not always get this
        else if (first == "VmSwap:") {
            swp = strTo<long long>(second);
        }
    }
    statusStream.close();

    recordStat(StatKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_VM_RSS_KB), rss);
    recordStat(StatKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_VM_SWAP_KB), swp);
    recordStat(StatKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_REAL_RAM_KB), rss+swp);
}

void DcgmMetadataManager::retrieveOSInfoFromProcSelfStat()
{
    dcgmReturn_t status;

    // parse /proc/self/stat for this process's CPU usage
    std::ifstream statStream("/proc/self/stat");

    std::vector<std::string> stats;
    std::string statVal;
    while (statStream >> statVal)
    {
        stats.push_back(statVal);
    }
    statStream.close();

    if (stats.size() <= 14)
    {
        PRINT_ERROR("", "could not retrieve expected fields from /proc/self/stat");
        return;
    }

    // use indices from "man 5 proc" section "/proc/[pid]/stat" minus 1 (they start at 1 instead of 0)
    // user and system (kernel) time of the process in clock ticks
    unsigned long utimeProc = strTo<unsigned long>(stats.at(13));
    unsigned long stimeProc = strTo<unsigned long>(stats.at(14));

    long long totProcTicks = utimeProc + stimeProc;

    StatKey uTimeKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_TICKS_UTIME);
    StatKey sTimeKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_TICKS_STIME);
    recordStat(uTimeKey, (long long)utimeProc);
    recordStat(sTimeKey, (long long)stimeProc);
}

void DcgmMetadataManager::retrieveOSInfoFromProcStat()
{
    std::ifstream sysStatStream("/proc/stat");
    std::string line;

    long long totDevTicks = 0;

    while (getline(sysStatStream, line))
    {
        std::stringstream ss(line);
        std::string entry;

        ss >> entry;

        if (entry == "cpu")
        {
            // sum all the different break-downs of cpu time
            unsigned long long ticks;
            while (ss >> ticks)
            {
              totDevTicks += ticks;
            }
        }
    }
    sysStatStream.close();

    StatKey sKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_DEVICE_TICKS_TOTAL);
    recordStat(sKey, totDevTicks);
}

// it is assumed that a stat lock is held when calling this function
template <typename T>
dcgmReturn_t DcgmMetadataManager::getFullStatDiffForType(StatKey sKey, T *dStat)
{
    dcgmReturn_t status;

    mcollect_value_p measurement = getStatMeasurement(sKey);

    if (sKey.cKey.context == STAT_CONTEXT_PROCESS)
    {
        status = validateMcollectType(measurement, sKey.mType.process);
    }
    else
    {
        status = validateMcollectType(measurement, sKey.mType.field);
    }

    if (DCGM_ST_OK != status)
    {
        return status;
    }

    status = getFullStatDiff(measurement->val.tseries, dStat);
    if (DCGM_ST_NO_DATA == status)
    {
        PRINT_DEBUG("%s", "metadata has not been gathered for long enough to take an average for stat %s",
                    sKey.str().c_str());
    }

    return status;
}

mcollect_value_p DcgmMetadataManager::getStatMeasurement(StatKey sKey)
{
    if (sKey.cKey.isGlobalStat())
    {
        return m_statCollection->GetGlobalStat(sKey.str());
    }
    else if (sKey.cKey.isGpuStat())
    {
        return m_statCollection->GetGpuStat(m_cacheManager->GpuIdToLwmlIndex(sKey.cKey.gpuId), sKey.str());
    }
    else
    {
        PRINT_ERROR("%s", "%s is an invalid context", sKey.cKey.str().c_str());
        return NULL;
    }
}

/**
 * Should run after "generateProcessInfo" so that required fields are gathered recently
 */
void DcgmMetadataManager::generateCpuUtilization()
{
    dcgmReturn_t status;

    long long dProcTicksUtime;
    long long dProcTicksStime;
    long long dDevTicks;

    StatKey uTimeProcKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_TICKS_UTIME);
    StatKey sTimeProcKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_TICKS_STIME);
    StatKey totalTimeDevKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_DEVICE_TICKS_TOTAL);

    {
        // hold a lock for this whole block
        DcgmLockGuard lock(m_mutex);

        status = getFullStatDiffForType(uTimeProcKey, &dProcTicksUtime);
        if (DCGM_ST_OK != status)
        {
            return;
        }

        status = getFullStatDiffForType(sTimeProcKey, &dProcTicksStime);
        if (DCGM_ST_OK != status)
        {
            return;
        }

        status = getFullStatDiffForType(totalTimeDevKey, &dDevTicks);
        if (DCGM_ST_OK != status)
        {
            return;
        }
    }

    if (dDevTicks == 0)
    {
        PRINT_ERROR("", "somehow got no change in device ticks over sampling interval");
        return;
    }

    StatKey uTimeCpuUtilKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_CPU_UTIL_UTIME);
    StatKey sTimeCpuUtilKey(ContextKey(STAT_CONTEXT_PROCESS), PROCESS_MT_CPU_UTIL_STIME);

    double recentCpuUtilUtime = (double)dProcTicksUtime / (double)dDevTicks;
    double recentCpuUtilStime = (double)dProcTicksStime / (double)dDevTicks;
    recordStat(uTimeCpuUtilKey, recentCpuUtilUtime);
    recordStat(sTimeCpuUtilKey, recentCpuUtilStime);
}

/**
 * This function is a critical section for the mcollect and should be locked when called.
 */
dcgmReturn_t DcgmMetadataManager::getFullStatDiff(timeseries_p ts,
                                                  long long *diff)
{
    timeseries_entry_p last = timeseries_last(ts, NULL);
    timeseries_entry_p first = timeseries_first(ts, NULL);

    if (NULL == last || NULL == first || first == last)
    {
        PRINT_DEBUG("", "not enough metadata has been gathered to callwlate a diff");
        return DCGM_ST_NO_DATA;
    }

    *diff = last->val.i64 - first->val.i64;
    return DCGM_ST_OK;
}
