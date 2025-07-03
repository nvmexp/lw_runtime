#ifndef DCGMMODULEINTROSPECT_H
#define DCGMMODULEINTROSPECT_H

#include "DcgmModule.h"
#include "DcgmMetadataMgr.h"
#include "dcgm_introspect_structs.h"

class DcgmModuleIntrospect : public DcgmModule
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleIntrospect();
    virtual ~DcgmModuleIntrospect(); /* Virtual because of ancient C++ library */

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     *
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
private:

    /*************************************************************************/
    /* Request Processesing helper methods */
    dcgmReturn_t GetMemUsageForFields(dcgmIntrospectContext_t *context,
                                      dcgmIntrospectFullMemory_t *memInfo,
                                      int waitIfNoData);
    dcgmReturn_t GetExecTimeForFields(dcgmIntrospectContext_t *context,
                                      dcgmIntrospectFullFieldsExecTime_t *execTime,
                                      int waitIfNoData);
    dcgmReturn_t GetMemUsageForHostengine(dcgmIntrospectMemory_t *memInfo, int waitIfNoData);
    dcgmReturn_t GetCpuUtilizationForHostengine(dcgmIntrospectCpuUtil_t *cpuUtil, int waitIfNoData);

    void CopyFieldsExecTime(dcgmIntrospectFieldsExecTime_t &execTime,
                            const DcgmMetadataManager::ExecTimeInfo &metadataExecTime);

    /*************************************************************************/
    /* Subrequest helpers
     */
    dcgmReturn_t ProcessMetadataFieldsExecTime(dcgm_introspect_msg_fields_exec_time_t *msg);
    dcgmReturn_t ProcessMetadataFieldsMemUsage(dcgm_introspect_msg_fields_mem_usage_t *msg);
    dcgmReturn_t ProcessMetadataHostEngineCpuUtil(dcgm_introspect_msg_he_cpu_util_t *msg);
    dcgmReturn_t ProcessMetadataHostEngineMemUsage(dcgm_introspect_msg_he_mem_usage_t *msg);
    dcgmReturn_t ProcessMetadataStateSetRunInterval(dcgm_introspect_msg_set_interval_t *msg);
    dcgmReturn_t ProcessMetadataStateToggle(dcgm_introspect_msg_toggle_t *msg);
    dcgmReturn_t ProcessMetadataUpdateAll(dcgm_introspect_msg_update_all_t *msg);

    /*************************************************************************/
    /*
     * Verify metadata gathering is enabled. Logs an error if it is not.
     */
    dcgmReturn_t VerifyMetadataEnabled();


    DcgmMetadataManager *mpMetadataManager; /* Pointer to the worker class for this module */
    DcgmCacheManager *mpCacheManager; /* Cached pointer to the cache manager. Not owned by this class */
};


#endif //DCGMMODULEINTROSPECT_H
