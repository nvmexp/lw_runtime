/* 
 * File:   LwcmHostEngineHandler.h
 */

#ifndef LWCMHOSTENGINEHANDLER_H
#define LWCMHOSTENGINEHANDLER_H

#include "dcgm_agent.h"
#include "LwcmServerConnection.h"
#include "LwcmCacheManager.h"
#include "LwcmProtobuf.h"
#include "LwcmGroup.h"
#include <iostream>
#include "DcgmModule.h"
#include "DcgmFieldGroup.h"
#include "DcgmWatcher.h"
#include "LwcmRequest.h"
#include <tr1/unordered_map>

using namespace std;

/* Module status structure */
typedef struct dcgmhe_module_info_t
{
    dcgmModuleId_t     id;     /* ID of this module  */
    dcgmModuleStatus_t status; /* Status of this module */
    DcgmModule         *ptr;   /* Pointer to the loaded class of this module */
    const char *filename;      /* Filename for this module like libdcgmmodulehealth.so */
    void *dlopenPtr;           /* Pointer to this loaded module returned by dlopen(). NULL if not loaded */
    dcgmModuleAlloc_f allocCB; /* Module function for allocating a DcgmModule object. NULL if not set */
    dcgmModuleFree_f freeCB;   /* Module function for freeing a DcgmModule object. NULL if not set */
} dcgmhe_module_info_t, *dcgmhe_module_info_p;

/*****************************************************************************
 This class extends LwcmServer and implements the OnRequest method to receive
 messages from the HeadNode.
 *****************************************************************************/
class LwcmHosEngineServer : public LwcmServer
{
public:
    /*****************************************************************************
     Constructor/Destructor
     *****************************************************************************/
    LwcmHosEngineServer(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP);
    virtual ~LwcmHosEngineServer();
    
    /*****************************************************************************
     This method handles the message received on the socket.

     This is a virtual method of LwcmHosEngineServer
     *****************************************************************************/
    int OnRequest(dcgm_request_id_t requestId, LwcmServerConnection *pConnection);

    /*****************************************************************************
     Notify this object that a connection was disconnected

     This is a virtual method of LwcmHosEngineServer
     *****************************************************************************/
    void OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);

    /*****************************************************************************
     * These methods are used to send response or push data to the client
     *****************************************************************************/
    int SendDataToClient(LwcmProtobuf *protoObj, LwcmServerConnection* pConnection, 
                         dcgm_request_id_t requestId, unsigned int msgType);
    
    dcgmReturn_t SendRawMessageToClient(dcgm_connection_id_t connectionId, 
                                        unsigned int msgType, dcgm_request_id_t requestId,
                                        void *msgData, int msgLength);
    
private:
    
};

typedef struct jobRecord_st
{
    unsigned int groupId;
    timelib64_t startTime;
    timelib64_t endTime;
}jobRecord_t;


class LwcmHostEngineHandler {
public:


    /*****************************************************************************
     * This method is used to initialize LWCM HostEngineHandler
     * @param mode
     * @return 
    *****************************************************************************/    
    static LwcmHostEngineHandler* Init(dcgmOperationMode_t mode);
    
    /*****************************************************************************
     This method is used to get Instance of HostEngine Handler.
     There will be just one instance for the node
     *****************************************************************************/
    static LwcmHostEngineHandler* Instance();
    
    /*****************************************************************************
     This method is used to run the server on host engine side.
     Must be ilwoked from HostEngine to start the server to listen for 
     connections.
     The corresponding "C" API will not be in the public header for the agent.
     The "C" API for this method will be part of Internal control APIs which can 
     be ilwoked by LW Host Engine
     *****************************************************************************/
    dcgmReturn_t RunServer(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP);
    
    /*****************************************************************************
     * This method is used to get instance of LWCM server
     *****************************************************************************/
    LwcmHosEngineServer * GetServer();
    
    
    /*****************************************************************************
     * This method is used to process one or more commands at the host engine. 
     * For Host Engine, this method is intended to be a common processing method 
     * for both Embedded and Stand-alone use case.
     * @param pVecCmdsToProcess : Serves as both Input and Output argument
     * @param pIsComplete       : Notifies if the command is complete with its processing
     * @param pConnection       : Pointer to the actual connection object this request came
     *                            from or NULL if this is an embedded request
     * @param requestId         : ID of the request this came from
     * 
     *
     * @return
     * 0        On Success
     * <0       On Error
     *****************************************************************************/
    int HandleCommands(vector<lwcm::Command *> *pVecCmdsToProcess, bool *pIsComplete, 
                       LwcmServerConnection* pConnection, 
                       dcgm_request_id_t requestId);

    /*****************************************************************************
     * This method is used to handle a client disconnecting from the host engine
     *****************************************************************************/
     void OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);

    /*****************************************************************************
     This method retrieves the instance of the current cache manager
     *****************************************************************************/
    DcgmCacheManager * GetCacheManager() { return mpCacheManager; }
    
    /*****************************************************************************
     This method retrieves the instance of the group manager
     *****************************************************************************/
    LwcmGroupManager * GetGroupManager() { return mpGroupManager; }
    
    /*****************************************************************************
     This method retrieves the instance of the field group manager
     *****************************************************************************/
    DcgmFieldGroupManager * GetFieldGroupManager() { return mpFieldGroupManager; }

    /*****************************************************************************/
    dcgmReturn_t ProcessModuleCommand(dcgm_module_command_header_t *moduleCommand);

    /*****************************************************************************
     This method is used to cleanup the Host Engine Handler Instance
     *****************************************************************************/
    void Cleanup();

    /*****************************************************************************
     * This method is used to get GPU Ids corresponding to all the devices on
     * the node. The GPU Ids are valid for a life-span of hostengine and cannot
     * be assumed to get same value across the reboots.
     *****************************************************************************/
    dcgmReturn_t  GetLwcmGpuIds(std::vector<unsigned int> &gpuIds,
                                int onlySupported);

    dcgmReturn_t  GetLwcmGpuArch(dcgm_field_eid_t entityId,
                                 lwmlChipArchitecture_t &arch);

    /*****************************************************************************
     * Process a WATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 unsigned short dcgmFieldId,
                                 timelib64_t monitorFrequencyUsec,
                                 double maxSampleAge,
                                 int maxKeepSamples,
                                 DcgmWatcher watcher);

    /*****************************************************************************
     * Process an UNWATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   unsigned short dcgmFieldId,
                                   int clearCache,
                                   DcgmWatcher watcher);

    /****************************************************************************
     * Get the most recent sample of a field
     *****************************************************************************/

    dcgmReturn_t GetLatestSample(dcgm_field_entity_group_t entityGroupId, 
                                 dcgm_field_eid_t entityId,
                                 unsigned short dcgmFieldId,
                                 dcgmcm_sample_p sample);

    /*****************************************************************************
     Notify this object that a group was removed from the group manager
     *****************************************************************************/
    void OnGroupRemove(unsigned int groupId);

    /*****************************************************************************
     Notify this object that field values we subscribed for updated.
     *****************************************************************************/
    void OnFvUpdates(DcgmFvBuffer *fvBuffer, DcgmWatcherType_t *watcherTypes, 
                     int numWatcherTypes, void *userData);

    /*****************************************************************************
     * Add a watcher to a local request. This watcher will be assigned a requestId
     * and will receive a ProcessMessage() call every time a message is sent from
     * the host engine to connectionId 0 (local) with its request ID. 
     * 
     * Note that request->requestId will be assigned by this call.
     * 
     * request    IN: A LwcmRequest instance that was allocated with new and will now
     *                belong to the host engine. You should not reference this object
     *                after this call.
     * requestId OUT: The request ID that was assigned to this request on success.
     * 
     *****************************************************************************/
    dcgmReturn_t AddRequestWatcher(LwcmRequest *request, dcgm_request_id_t *requestId);

    /*****************************************************************************
     * Notify a DcgmRequest object that it has received its last response and
     * thus should be cleaned up by its owner.
     */
    void NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, 
                                   dcgm_request_id_t requestId);

    /*****************************************************************************
     * Send a raw message to a connected client
     * 
     *****************************************************************************/
    dcgmReturn_t SendRawMessageToClient(dcgm_connection_id_t connectionId, 
                                        unsigned int msgType, dcgm_request_id_t requestId,
                                        void *msgData, int msgLength);
    dcgmReturn_t SendRawMessageToEmbeddedClient(unsigned int msgType, 
                                                dcgm_request_id_t requestId,
                                                void *msgData, int msgLength);

private:
    LWOSCriticalSection m_lock; /* Lock used for accessing table of job stats
                                   and the objects within them */

    /**************************************************************************
    * Lock/Unlocks methods
    **************************************************************************/
    int Lock();
    int Unlock();

    /*****************************************************************************
     * This method is used to process a single command on the host engine
     *****************************************************************************/
    int ProcessRequest(lwcm::Command *pCmd, bool *pIsComplete, 
                       LwcmServerConnection* pConnection, 
                       dcgm_request_id_t requestId);    

    /*****************************************************************************
     This method is used to serialize the cache manager to a file
    *****************************************************************************/
    dcgmReturn_t SaveCachedStats(lwcm::CacheManagerSave *cacheManagerSave);

    /*****************************************************************************
     Deletes an object if it is not null then sets its pointer to null
     *****************************************************************************/
    template<typename T>
    static void deleteNotNull(T *&obj);

    /*****************************************************************************
     This method is used to deserialize the cache manager from a file
    *****************************************************************************/
    dcgmReturn_t LoadCachedStats(lwcm::CacheManagerLoad *cacheManagerLoad);
    
    /*****************************************************************************
     * This method is used to get GPU Ids corresponding to all the devices on
     * the node. The GPU Ids are valid for a life-span of hostengine and cannot
     * be assumed to get same value across the reboots.
     *****************************************************************************/
    dcgmReturn_t GetLwcmGpuIds(lwcm::FieldMultiValues *pLwcmFieldMultiValues, int onlySupported);

    /*****************************************************************************
    * This method is used to query Cache Manager to get latest sample for a field 
    *****************************************************************************/
    dcgmReturn_t GetFieldValue(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId, unsigned int fieldId,
                               lwcm::FieldValue* pLwcmFieldValue);
    
    /*****************************************************************************
     * This method is used to get values corresponding to the fields
     *****************************************************************************/
    dcgmReturn_t GetValuesForFields(dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId, unsigned int fieldIds[],
                                    unsigned int count, lwcm::FieldValue values[]);

    /*****************************************************************************
     * This method is used to inject a field value
     *****************************************************************************/
    dcgmReturn_t InjectFieldValue(dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId,
                                  lwcm::InjectFieldValue *injectFieldValue);

    /*****************************************************************************
     * This method is get information for a field in the cache manager
     *****************************************************************************/
    dcgmReturn_t GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo);

    /*****************************************************************************
     * Process a WATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                 dcgm_field_eid_t entityId,
                                 const lwcm::WatchFieldValue *watchFieldValue,
                                 DcgmWatcher watcher);

    /*****************************************************************************
     * Process an UNWATCH_FIELD_VALUE message
     *****************************************************************************/
    dcgmReturn_t UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                   dcgm_field_eid_t entityId,
                                   const lwcm::UnwatchFieldValue *unwatchFieldValue,
                                   DcgmWatcher watcher);

    /*****************************************************************************
     * Process an UPDATE_ALL_FIELDS message
     *****************************************************************************/
    dcgmReturn_t UpdateAllFields(const lwcm::UpdateAllFields *updateAllFields);    

    /*****************************************************************************
     * Process an GET_FIELD_MULTIPLE_VALUES message
     *****************************************************************************/
    dcgmReturn_t GetFieldMultipleValues(dcgm_field_entity_group_t entityGroupId,
                                        dcgm_field_eid_t entityId,
                                        lwcm::FieldMultiValues *pFieldMultiValues);

    /*****************************************************************************
     * Process a GET_PROCESS_INFO message
     *****************************************************************************/
    dcgmReturn_t GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo);
    
    /*****************************************************************************/
    dcgmReturn_t JobStartStats(string jobId, unsigned int groupId);
    
    /*****************************************************************************/
    dcgmReturn_t JobStopStats(string jobId);
    
    /*****************************************************************************/
    dcgmReturn_t JobGetStats(string jobId, dcgmJobInfo_t* pJobInfo);

    /*****************************************************************************/
    dcgmReturn_t JobRemove(string jobId);

    /*****************************************************************************/
    dcgmReturn_t JobRemoveAll(void);

    /*****************************************************************************
     * This method is used to try to load a module of DCGM
     * 
     * Returns DCGM_ST_OK on success or if the module is already loaded
     *         DCGM_ST_MODULE_NOT_LOADED on error
     *****************************************************************************/
    dcgmReturn_t LoadModule(dcgmModuleId_t moduleId);

    /*****************************************************************************/
    /* Helper methods */
    dcgmReturn_t  HelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId, unsigned short fieldId,
                                            dcgmStatSummaryInt64_t *summary,
                                            long long startTime, long long endTime);
    dcgmReturn_t  HelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId, unsigned short fieldId,
                                            dcgmStatSummaryInt32_t *summary,
                                            long long startTime, long long endTime);

    /*****************************************************************************
     * Add a watch on a field group
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t WatchFieldGroup(unsigned int groupId, dcgmFieldGrp_t fieldGroupId,
                                 timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                 int maxKeepSamples, DcgmWatcher watcher);

    /*****************************************************************************
     * Remove a watch on a field group
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
     dcgmReturn_t UnwatchFieldGroup(unsigned int groupId, dcgmFieldGrp_t fieldGroupId,
                                    DcgmWatcher watcher);

    /*****************************************************************************
     * Add a watch on a field group for all GPUs
     *
     * activeOnly: Whether or not to only watch the field group on GPUs that
     *             are active. Inactive GPUs include GPUs that are not whitelisted
     *
     * This helper is used both internally and externally
     *
     ****************************************************************************/
    dcgmReturn_t WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                        timelib64_t monitorFrequencyUsec, double maxSampleAge,
                                        int maxKeepSamples, int activeOnly, DcgmWatcher watcher);

    /*****************************************************************************
     Helper functions for the scheduler hint API
     *****************************************************************************/
    dcgmReturn_t TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds);

    void RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds);

    dcgmReturn_t ProcessSelectGpusByTopology(lwcm::Command *pCmd, bool *pIsComplete);

    /*****************************************************************************
     Helper method to RPC to the health module for a health check
     *****************************************************************************/
    dcgmReturn_t HelperHealthCheckV1(unsigned int groupId, 
                                     long long startTime, 
                                     long long endTime, 
                                     dcgmHealthResponse_v1 *response);



    /*****************************************************************************
     Helper method for watching the fields that the host engine cares about
     *****************************************************************************/
    dcgmReturn_t WatchHostEngineFields(void);

    dcgmReturn_t ProcessClientLogin(lwcm::Command *pCmd, bool *pIsComplete, LwcmServerConnection *pConnection);
    dcgmReturn_t ProcessGroupCreate(lwcm::Command *pCmd, bool *pIsComplete, LwcmServerConnection *pConnection,
            dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessAddRemoveGroup(lwcm::Command *pCmd, bool *pIsComplete, LwcmServerConnection *pConnection,
            dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupDestroy(lwcm::Command *pCmd, bool *pIsComplete, LwcmServerConnection *pConnection,
            dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupInfo(lwcm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessGroupGetallIds(lwcm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t connectionId);
    dcgmReturn_t ProcessDiscoverDevices(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetEntityList(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessRegForPolicyUpdate(lwcm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t pConnectionId,
                                  LwcmServerConnection* pConnection, dcgm_request_id_t requestId);
    dcgmReturn_t ProcessUnregForPolicyUpdate(lwcm::Command *pCmd, bool *pIsComplete, LwcmServerConnection *pConnection);
    dcgmReturn_t ProcessSetLwrrentViolPolicy(lwcm::Command *pCmd, bool *pIsComplete, dcgm_connection_id_t pConnectionId);
    dcgmReturn_t ProcessGetLwrrentViolPolicy(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessSaveCachedStats(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessLoadCachedStats(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessInjectFieldValue(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetFieldLatestValue(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetFieldMultipleValues(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessWatchFieldValue(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessUnwatchFieldValue(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessUpdateAllFields(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessCacheManagerFieldInfo(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessWatchFields(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessUnwatchFields(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessGetPidInfo(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessFieldGroupCreate(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessFieldGroupDestroy(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessFieldGroupGetOne(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessFieldGroupGetAll(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessWatchRedefined(lwcm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher);
    dcgmReturn_t ProcessJobStartStats(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobStopStats(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobRemove(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessJobGetInfo(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetTopologyAffinity(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetTopologyIO(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessModuleCommandWrapper(lwcm::Command *pCmd, bool *pIsComplete, 
                                             dcgm_connection_id_t connectionId, dcgm_request_id_t requestId);
    dcgmReturn_t ProcessCreateFakeEntities(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetLwLinkLinkStatus(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetMultipleLatestValues(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessSetLwLinkLinkStatus(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessGetFieldSummary(lwcm::Command *pCmd, bool *pIsComplete);
    dcgmReturn_t ProcessModuleBlacklist(lwcm::Command *pCmd);
    dcgmReturn_t ProcessModuleGetStatuses(lwcm::Command *pCmd);

    /*****************************************************************************/
    /* Remove any requests that the host engine was tracking */
    dcgmReturn_t RemoveAllTrackedRequests(void);

    /*****************************************************************************
     Private Constructor and Destructor to achieve Singelton design
     *****************************************************************************/
    LwcmHostEngineHandler() {}
    LwcmHostEngineHandler(dcgmOperationMode_t mode);
    virtual ~LwcmHostEngineHandler();

    /* This data structure is used to store user provided job id information and associates start
       and stop timestamp with the user provided start/stop notification. */
    typedef map<string, jobRecord_t> jobIdMap_t;
    jobIdMap_t mJobIdMap;    
    
    LwcmHosEngineServer *mpServerObj;                              // Host Engine Server Object
    static LwcmHostEngineHandler * mpHostEngineHandlerInstance;    // HostEngine Handler Instance
    DcgmCacheManager *mpCacheManager;
    LwcmGroupManager  *mpGroupManager;
    DcgmFieldGroupManager *mpFieldGroupManager;

    /* Field Groups */
    dcgmFieldGrp_t mFieldGroup1Sec;
    dcgmFieldGrp_t mFieldGroup30Sec;
    dcgmFieldGrp_t mFieldGroupHourly;
    dcgmFieldGrp_t mFieldGroupPidAndJobStats;

    void HandleAddWatchError(int ret, std::string field);
    void finalizeCmd(lwcm::Command *pCmd, dcgmReturn_t cmdStatus,
                     bool *&pIsComplete, void* returnArg, size_t returnArgSize);

    /* This data structure stores pluggable modules for handling client requests */
    dcgmhe_module_info_t m_modules[DcgmModuleIdCount];

    /* Watched requests. Lwrrently used to track policy management callbacks. Protected by Lock()/Unlock() */
    dcgm_request_id_t m_nextWatchedRequestId;
    typedef std::tr1::unordered_map<dcgm_request_id_t, LwcmRequest*> watchedRequests_t;
    watchedRequests_t m_watchedRequests;
};

#endif /* LWCMHOSTENGINEHANDLER_H */
