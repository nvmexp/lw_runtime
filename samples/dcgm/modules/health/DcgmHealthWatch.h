#ifndef _DCGM_HEALTH_WATCH_H
#define _DCGM_HEALTH_WATCH_H

#include "LwcmGroup.h"
#include "dcgm_agent_internal.h"
#include "LwcmCacheManager.h"
#include "DcgmGPUHardwareLimits.h"
#include "DcgmError.h"


// accomodate values for SBE/DBE as well as retired and pending retired pages
#define DCGM_HEALTH_MEMORY_NUM_FIELDS 5

//Number of lwlink error counter types
#define DCGM_HEALTH_WATCH_LWLINK_ERROR_NUM_FIELDS LWML_LWLINK_ERROR_COUNT

/* This class is implements the background health check methods
 * within the hostengine
 * It is intended to set watches, monitor them on demand, and 
 * inform a user of any specific problems for watches that have
 * been requested.
 */
class DcgmHealthWatch
{
public:
    DcgmHealthWatch(LwcmGroupManager *gm, DcgmCacheManager *cm);
    ~DcgmHealthWatch();

    /*
     * This method is used to set the watches based on which bits are enabled in the 
     * systems variable 
     */
    dcgmReturn_t SetWatches(unsigned int groupId, dcgmHealthSystems_t systems,
                            dcgm_connection_id_t connectionId);

    /*
     * This method is used to get the watches 
     */
    dcgmReturn_t GetWatches(unsigned int groupId, dcgmHealthSystems_t *systems);

    /*
     * This method is used to check an individual gpu's health watches
     */
    dcgmReturn_t MonitorWatchesForGpu(unsigned int gpuId, long long startTime, long long endTime,
                                      dcgmHealthSystems_t healthResponse, dcgmHealthResponse_v1 *response);

    /*
     * This method is used to trigger a monitoring of the configured watches for a group
     */
    dcgmReturn_t MonitorWatchesV1(unsigned int groupId, long long startTime, long long endTime, dcgmHealthResponse_v1 *response);
    dcgmReturn_t MonitorWatchesV2(unsigned int groupId, long long startTime, long long endTime, void *response);

    /*
     Notify this module that a group was removed from the group manager
     */
    void OnGroupRemove(unsigned int groupId);

private:
    LwcmGroupManager *mpGroupManager;
    DcgmCacheManager *mpCacheManager;
    /* Map of groupId -> dcgmHealthSystems_t of the watched health systems of a given groupId */
    typedef std::map<unsigned int, dcgmHealthSystems_t>groupWatchTable_t;
    groupWatchTable_t mGroupWatchState;

    DcgmMutex *m_mutex;

    /* Prepopulated lists of fields used by various internal methods */
    std::vector<unsigned int> m_lwSwitchNonFatalFieldIds; /* LwSwitch non-fatal errors */
    std::vector<unsigned int> m_lwSwitchFatalFieldIds;    /* LwSwitch fatal errors */

    enum LwcmMemoryErrorLocations_enum
    {
        DCGM_HEALTH_MEMORY_SBE_VOL_TOTAL  = 0,
        DCGM_HEALTH_MEMORY_DBE_VOL_TOTAL  = 1,
        DCGM_HEALTH_MEMORY_LOC_COUNT
    };

    enum LwcmMemoryRetiredPages_enum
    {
        DCGM_HEALTH_MEMORY_PAGE_RETIRED_SBE = 2, /* Should start from DCGM_HEALTH_MEMORY_LOC_COUNT */
        DCGM_HEALTH_MEMORY_PAGE_RETIRED_DBE = 3,
        DCGM_HEALTH_MEMORY_PAGE_RETIRED_PENDING = 4,
    };

    // DCGM internal callback table for the engine
    etblDCGMEngineInternal *m_etblLwcmEngineInternal;

    // miscellaneous helper methods
    std::string MemFieldToString(unsigned short fieldId);

    /* Build internal lists of fieldIds to be used by other methods */
    void BuildFieldLists(void);

    void SetResponse(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, dcgmHealthWatchResults_t status, dcgmHealthSystems_t system, 
                DcgmError &d, void *response, bool newIncidentRecord);
    void SetResponseV1(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, dcgmHealthWatchResults_t status, dcgmHealthSystems_t system, 
                const char *description, dcgmHealthResponse_v1 *response, bool newIncidentRecord);
    void SetResponseV2(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, dcgmHealthWatchResults_t status, dcgmHealthSystems_t system, 
                const char *description, dcgmHealthResponse_v2 *response, bool newIncidentRecord);
    void SetResponseV3(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,
                       dcgmHealthWatchResults_t status, dcgmHealthSystems_t system, const DcgmError &d,
                       dcgmHealthResponse_v3 *response, bool newIncidentRecord);

    void RecordNewIncidentV1(dcgmHealthResponse_v1 *response, unsigned int gpuIndex,
                             dcgmHealthWatchResults_t status, dcgmHealthSystems_t system,
                             const char *description);
    void RecordNewIncidentV2(dcgmHealthResponse_v2 *response, unsigned int entityIndex,
                             dcgmHealthWatchResults_t status, dcgmHealthSystems_t system,
                             const char *description);

    void RecordNewIncidentV3(dcgmHealthResponse_v3 *response, unsigned int entityIndex,
                             dcgmHealthWatchResults_t status, dcgmHealthSystems_t system, const DcgmError &d);

    void RecordErrorV3(dcgmHealthResponse_v3 *response, unsigned int entityIndex, unsigned int incidentIndex,
                       const DcgmError &d);

    // methods to handle setting watches
    dcgmReturn_t SetPcie(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetMem(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetInforom(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetThermal(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetPower(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetLWLink(unsigned int gpuId, bool enable, DcgmWatcher watcher);
    dcgmReturn_t SetLwSwitchWatches(std::vector<unsigned int> &groupSwitchIds,
                                    dcgmHealthSystems_t systems,
                                    DcgmWatcher watcher);

    dcgmReturn_t MonitorPcie(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorMem(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId,long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorInforom(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorThermal(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorPower(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorLWLink(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, long long startTime, long long endTime, void *response);
    dcgmReturn_t MonitorLwSwitchErrorCounts(bool fatal, dcgm_field_entity_group_t entityGroupId, 
                                            dcgm_field_eid_t entityId, long long startTime, 
                                            long long endTime, void *response);
};

#endif //_DCGM_HEALTH_WATCH_H
