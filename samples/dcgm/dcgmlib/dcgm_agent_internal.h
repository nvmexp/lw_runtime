/* 
 * File:   dcgm_agent_internal.h
 */

#ifndef DCGM_AGENT_INTERNAL_H
#define	DCGM_AGENT_INTERNAL_H

#ifdef	__cplusplus
extern "C" {
#endif

#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_uuid.h"    
#include <stdint.h>
    
/*****************************************************************************
 *****************************************************************************/
/*****************************************************************************
 * Host Engine Internal Methods to be used by LW Tools (Used by LW HostEngine)
 *****************************************************************************/
/*****************************************************************************
 *****************************************************************************/

#define DCGM_EMBEDDED_HANDLE 0x7fffffff
    
    
    
// GUIDS for internal APIs
DCGM_DEFINE_UUID(ETID_DCGMEngineInternal,
                 0x7c3efec4, 0x9fc9, 0x5e6c, 0xb3, 0x37, 0xfe, 0x79, 0x7e, 0x22, 0xe7, 0xd4);

typedef struct etblDCGMEngineInternal_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;
    
    /**
     * This method starts the Host Engine Server
     *
     * portNumber      IN: TCP port to listen on. This is only used if isTcp == 1.
     * socketPath      IN: This is the path passed to bind() when creating the socket
     *                     For isConnectionTCP == 1, this is the bind address. "" or NULL = All interfaces
     *                     For isConnectionTCP == 0, this is the path to the domain socket to use
     * isConnectionTCP IN: Whether to listen on a TCP/IP socket (1) or a unix domain socket (0)
     */
    dcgmReturn_t (*fpdcgmServerRun)(unsigned short portNumber, char *socketPath, unsigned int isConnectionTCP);

    /**
     * This method is used to get values corresponding to the fields. 
     * Return:
     * DCGM_ST_SUCCESS  On Success. Even when the API returns success, check for
     *                  individual status inside each field. 
     *                  Look at values[index].status. The field values will be
     *                  populated only when status in each field is DCGM_ST_SUCCESS
     * DCGM_ST_?        In case of error
     */
    dcgmReturn_t (*fpdcgmGetLatestValuesForFields)(dcgmHandle_t pLwcmHandle, int gpuId, unsigned short fieldIds[],
                                 unsigned int count, dcgmFieldValue_v1 values[]);

    /**
     * This method is used to get multiple values for a single field
     *
     * Returns DCGM_ST_SUCCESS on success.
     *         Other DCGM_ST_? error code on failure
     *
     */
    dcgmReturn_t (*fpdcgmGetMultipleValuesForField)(dcgmHandle_t pLwcmHandle, int gpuId, unsigned short fieldId,
                         int *count, long long startTs, long long endTs,
                         dcgmOrder_t order, dcgmFieldValue_v1 values[]);

    /**
     * Request updates for all field values that have updated since a given timestamp
     *
     * @param groupId             IN: Group ID representing a collection of one or more GPUs
     *                                Refer to \ref dcgmEngineGroupCreate for details on creating a group
     * @param sinceTimestamp      IN: Timestamp to request values since in usec since 1970. This will
     *                                be returned in nextSinceTimestamp for subsequent calls
     *                                0 = request all data
     * @param fieldIds            IN: Fields to return data for
     * @param numFieldIds         IN: Number of entries in fieldIds
     * @param nextSinceTimestamp OUT: Timestamp to use for sinceTimestamp on next call to this function
     * @param enumCB              IN: Callback to ilwoke for every field value update. Note that
     *                                multiple updates can be returned in each invocation
     * @param userData            IN: User data pointer to pass to the userData field of enumCB.
     */
    dcgmReturn_t (*fpdcgmGetFieldValuesSince)(dcgmHandle_t pLwcmHandle, dcgmGpuGrp_t groupId, long long sinceTimestamp,
                                                    unsigned short *fieldIds, int numFieldIds,
                                                    long long *nextSinceTimestamp,
                                                    dcgmFieldValueEnumeration_f enumCB, void *userData);

    /**
     * This method is used to tell the cache manager to watch a field value
     *
     * @param gpuId                               GPU ID to watch field on
     * @param fieldId                             Field ID to watch
     * @param updateFreq                          How often to update this field in usec
     * @param maxKeepAge                          How long to keep data for this field in seconds
     * @param maxKeepSamples                      Maximum number of samples to keep. 0=no limit
     *
     * @return
     *        - \ref DCGM_ST_OK                   if the call was successful
     *        - \ref DCGM_ST_BADPARAM             if \a gpuId, \a fieldId, \a updateFreq, \a maxKeepAge,
     *                                            or \a maxKeepSamples are invalid
     */
    dcgmReturn_t (*fpdcgmWatchFieldValue)(dcgmHandle_t pLwcmHandle, int gpuId, unsigned short fieldId,
                                              long long updateFreq, double maxKeepAge, int maxKeepSamples);

    /**
     * This method is used to tell the cache manager to unwatch a field value
     *
     * @param gpuId                               GPU ID to watch field on
     * @param fieldId                             Field ID to watch
     * @param clearCache                          Whether or not to clear all cached data for
     *                                            the field after the watch is removed
     * @return
     *        - \ref DCGM_ST_OK                   if the call was successful
     *        - \ref DCGM_ST_BADPARAM             if \a gpuId, \a fieldId, or \a clearCache is invalid
     */
    dcgmReturn_t (*fpdcgmUnwatchFieldValue)(dcgmHandle_t pLwcmHandle, int gpuId, unsigned short fieldId, int clearCache);


    /**
     * Deprecated
     *
     */
    void* unused;
    /**
     * Get the current amount of memory used to store the given field.
     *
     * @param pDcgmHandle                   IN: DCGM Handle
     * @param fieldId                       IN: DCGM field ID
     * @param pDcgmMetadataMemory          OUT: Total memory usage information for all field values in DCGM
     * @param waitIfNoData                  IN: if no metadata is gathered wait till this oclwrs (!0)
 *                                              or return DCGM_ST_NO_DATA (0)
     * @return
     *        - \ref DCGM_ST_OK                   if the call was successful
     *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \ref DCGM_METADATA_STATE_DISABLED
     *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
     */
    dcgmReturn_t (*fpdcgmIntrospectGetFieldMemoryUsage)(dcgmHandle_t pDcgmHandle,
                                                        unsigned short fieldId,
                                                        dcgmIntrospectFullMemory_t *memoryInfo,
                                                        int waitIfNoData);

    /*************************************************************************/
    /*
     * Set the interval (in milliseconds) for when the metadata manager should
     * do its collection runs.
     *
     * @param pDcgmHandle             IN: DCGM Handle
     * @param runIntervalMs          OUT: interval duration to set
     */
    dcgmReturn_t (*fpdcgmMetadataStateSetRunInterval)(dcgmHandle_t pDcgmHandle, unsigned int runIntervalMs);

    /*************************************************************************/
    /**
     * Get the total exelwtion time since startup used to update a field in DCGM.
     *
     * @param pDcgmHandle                   IN: DCGM Handle
     * @param fieldId                       IN: field ID
     * @param execTime                     OUT: see \ref dcgmFieldExecTime_t
     * @param waitIfNoData                  IN: if no metadata is gathered wait till this oclwrs (!0)
     *                                          or return DCGM_ST_NO_DATA (0)
     * @return
     *        - \ref DCGM_ST_OK                   if the call was successful
     *        - \ref DCGM_ST_NOT_CONFIGURED       if metadata gathering state is \ref DCGM_METADATA_STATE_DISABLED
     *        - \ref DCGM_ST_NO_DATA              if \a waitIfNoData is false and metadata has not been gathered yet
     */
    dcgmReturn_t (*fpdcgmIntrospectGetFieldExecTime)(dcgmHandle_t pDcgmHandle,
                                                     unsigned short fieldId,
                                                     dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                     int waitIfNoData);
    void * placeholder;

    /*************************************************************************/
    /**
     * Used to set vGPU configuration for the group of one or more GPUs identified by \a groupId.
     *
     * The configuration settings specified in \a pDeviceConfig are applied to all the GPUs in the
     * group. Since DCGM groups are a logical grouping of GPUs, the configuration settings Set for a group
     * stay intact for the individual GPUs even after the group is destroyed.
     *
     * If the user wishes to ignore the configuration of one or more properties in the input
     * \a pDeviceConfig then the property should be specified as one of \a DCGM_INT32_BLANK,
     * \a DCGM_INT64_BLANK, \a DCGM_FP64_BLANK or \a DCGM_STR_BLANK based on the data type of the
     * property to be ignored.
     *
     * If any of the properties fail to be configured for any of the GPUs in the group then the API
     * returns an error. The status handle \a statusHandle should be further evaluated to access error
     * attributes for the failed operations. Please refer to status management APIs at \ref DCGMAPI_ST
     * to access the error attributes.

     * @param pDcgmHandle           IN  : DCGM Handle
     *
     * @param groupId               IN  :  Group ID representing collection of one or more GPUs. Look
     *                                     at \ref dcgmGroupCreate for details on creating the
     *                                     group.
     * @param pDeviceConfig         IN  :  Pointer to memory to hold desired configuration to be
     *                                     applied for all the GPU in the group represented by
     *                                     \a groupId. The caller must populate the version field of
     *                                     \a pDeviceConfig.
     * @param statusHandle       IN/OUT :  Resulting error status for multiple operations. Pass it as
     *                                     NULL if the detailed error information is not needed.
     *                                     Look at \ref dcgmStatusCreate for details on creating
     *                                     status handle.

     * @return
     *        - \ref DCGM_ST_OK                   if the configuration has been successfully set.
     *        - \ref DCGM_ST_BADPARAM             if any of \a groupId or \a pDeviceConfig is invalid.
     *        - \ref DCGM_ST_VER_MISMATCH         if \a pDeviceConfig has the incorrect version.
     *        - \ref DCGM_ST_GENERIC_ERROR        if an unknown error has oclwrred.
     */
    dcgmReturn_t (*fpdcgmVgpuConfigSet)(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmVgpuConfig_t *pDeviceConfig,
                                                dcgmStatus_t statusHandle);

    /*************************************************************************/
    /**
     * Used to get vGPU configuration for all the GPUs present in the group.
     *
     * This API can get the most recent target or desired configuration set by \ref dcgmConfigSet.
     * Set type as \a DCGM_CONFIG_TARGET_STATE to get target configuration. The target configuration
     * properties are maintained by DCGM and are automatically enforced after a GPU reset or
     * reinitialization is completed.
     *
     * The method can also be used to get the actual configuration state for the GPUs in the group.
     * Set type as \a DCGM_CONFIG_LWRRENT_STATE to get the actually configuration state. Ideally, the
     * actual configuration state will be exact same as the target configuration state.
     *
     * If any of the property in the target configuration is unknown then the property value in the
     * output is populated as  one of DCGM_INT32_BLANK, DCGM_INT64_BLANK, DCGM_FP64_BLANK or
     * DCGM_STR_BLANK based on the data type of the property.
     *
     * If any of the property in the current configuration state is not supported then the property
     * value in the output is populated as one of DCGM_INT32_NOT_SUPPORTED, DCGM_INT64_NOT_SUPPORTED,
     * DCGM_FP64_NOT_SUPPORTED or DCGM_STR_NOT_SUPPORTED based on the data type of the property.
     *
     * If any of the properties can't be fetched for any of the GPUs in the group then the API returns
     * an error. The status handle \a statusHandle should be further evaluated to access error
     * attributes for the failed operations. Please refer to status management APIs at \ref DCGMAPI_ST
     * to access the error attributes.
     *
     * @param pDcgmHandle           IN  :  DCGM Handle
     * @param groupId               IN  :  Group ID representing collection of one or more GPUs. Look
     *                                     at \ref dcgmGroupCreate for details on creating the
     *                                     group.
     * @param type                  IN  :  Type of configuration values to be fetched.
     * @param count                 IN  :  The number of entries that \a deviceConfigList array can
     *                                     store.
     * @param deviceConfigList      OUT :  Pointer to memory to hold requested configuration
     *                                     corresponding to all the GPUs in the group (\a groupId). The
     *                                     size of the memory must be greater than or equal to hold
     *                                     output information for the number of GPUs present in the
     *                                     group (\a groupId).
     * @param statusHandle       IN/OUT :  Resulting error status for multiple operations. Pass it as
     *                                     NULL if the detailed error information is not needed.
     *                                     Look at \ref dcgmStatusCreate for details on creating
     *                                     status handle.

     * @return
     *        - \ref DCGM_ST_OK                   if the configuration has been successfully fetched.
     *        - \ref DCGM_ST_BADPARAM             if any of \a groupId, \a type, \a count,
     *                                            or \a deviceConfigList is invalid.
     *        - \ref DCGM_ST_NOT_CONFIGURED       if the target configuration is not already set.
     *        - \ref DCGM_ST_VER_MISMATCH         if \a deviceConfigList has the incorrect version.
     *        - \ref DCGM_ST_GENERIC_ERROR        if an unknown error has oclwrred.
     */
    dcgmReturn_t (*fpdcgmVgpuConfigGet)(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmConfigType_t type, int count,
                dcgmVgpuConfig_t deviceConfigList[], dcgmStatus_t statusHandle);

    /*************************************************************************/
    /**
     * Used to enforce previously set vGPU configuration for all the GPUs present in the group.
     *
     * This API provides a mechanism to the users to manually enforce the configuration at any point of
     * time. The configuration can only be enforced if it's already configured using the API \ref
     * dcgmConfigSet.
     *
     * If any of the properties can't be enforced for any of the GPUs in the group then the API returns
     * an error. The status handle \a statusHandle should be further evaluated to access error
     * attributes for the failed operations. Please refer to status management APIs at \ref DCGMAPI_ST
     * to access the error attributes.
     *
     * @param pDcgmHandle           IN  : DCGM Handle
     *
     * @param groupId               IN  :   Group ID representing collection of one or more GPUs. Look at
     *                                      \ref dcgmGroupCreate for details on creating the group.
     *                                      Alternatively, pass in the group id as \a DCGM_GROUP_ALL_GPUS
     *                                      to perform operation on all the GPUs.
     * @param statusHandle       IN/OUT :   Resulting error status for multiple operations. Pass it as
     *                                      NULL if the detailed error information is not needed.
     *                                      Look at \ref dcgmStatusCreate for details on creating
     *                                      status handle.
     * @return
     *        - \ref DCGM_ST_OK                   if the configuration has been successfully enforced.
     *        - \ref DCGM_ST_BADPARAM             if \a groupId is invalid.
     *        - \ref DCGM_ST_NOT_CONFIGURED       if the target configuration is not already set.
     *        - \ref DCGM_ST_GENERIC_ERROR        if an unknown error has oclwrred.
     */
    dcgmReturn_t (*fpdcgmVgpuConfigEnforce)(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId, dcgmStatus_t statusHandle);

    /*************************************************************************/
    /**
     * Gets vGPU device attributes corresponding to the \a gpuId. If operation is not successful for any of
     * the requested fields then the field is populated with one of DCGM_BLANK_VALUES defined in
     * dcgm_structs.h.
     *
     * @param pDcgmHandle   IN      : DCGM Handle
     * @param gpuId         IN      : GPU Id corresponding to which the attributes
     *                                should be fetched
     * @param pDcgmVgpuAttr IN/OUT  : vGPU Device attributes corresponding to \a gpuId.<br>
     *                                .version should be set to
     *                                \ref dcgmVgpuDeviceAttributes_version before this
     *                                call.
     *
     * @return
     *        - \ref DCGM_ST_OK            if the call was successful.
     *        - \ref DCGM_ST_VER_MISMATCH  if .version is not set or is invalid.
     */
    dcgmReturn_t (*fpdcgmGetVgpuDeviceAttributes)(dcgmHandle_t pDcgmHandle, unsigned int gpuId, dcgmVgpuDeviceAttributes_t *pDcgmVgpuAttr);

    /*************************************************************************/
    /**
     * Gets vGPU attributes corresponding to the \a vgpuId. If operation is not successful for any of
     * the requested fields then the field is populated with one of DCGM_BLANK_VALUES defined in
     * dcgm_structs.h.
     *
     * @param pDcgmHandle       IN          : DCGM Handle
     * @param vgpuId            IN          : vGPU Id corresponding to which the attributes
     *                                        should be fetched
     * @param pDcgmVgpuInstanceAttr IN/OUT  : vGPU attributes corresponding to \a vgpuId.<br>
     *                                        .version should be set to
     *                                        \ref dcgmVgpuInstanceAttributes_version before this
     *                                        call.
     *
     * @return
     *        - \ref DCGM_ST_OK            if the call was successful.
     *        - \ref DCGM_ST_VER_MISMATCH  if .version is not set or is invalid.
     */
    dcgmReturn_t (*fpdcgmGetVgpuInstanceAttributes)(dcgmHandle_t pDcgmHandle, unsigned int vgpuId, dcgmVgpuInstanceAttributes_t *pDcgmVgpuInstanceAttr);

    /**
     * Stop a diagnostic if there is one lwrrently running.
     * 
     * @param pDcgmHandle                   IN: DCGM Handle
     *                                           
     * @return 
     *        - \ref DCGM_ST_OK                   if the call was successful
     *        - \ref DCGM_ST_BADPARAM             if a provided parameter is invalid or missing
     */
    dcgmReturn_t (*fpdcgmStopDiagnostic)(dcgmHandle_t pDcgmHandle);

} etblDCGMEngineInternal;


// GUIDs for Internal testing support table
DCGM_DEFINE_UUID(ETID_DCGMEngineTestInternal,
                 0x8c4eabc6, 0x2ea8, 0x4e7d, 0xa3, 0x58, 0xef, 0x81, 0x4d, 0x21, 0xc3, 0xa5);

typedef struct etblDCGMEngineTestInternal_st {
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.    
    size_t struct_size;
    
    /**
     * This method injects a sample into the cache manager
     *
     * @param gpuId
     * @param dcgmInjectFieldValue
     */
    dcgmReturn_t (*fpdcgmInjectFieldValue)(dcgmHandle_t pLwcmHandle, unsigned int gpuId, dcgmInjectFieldValue_t *dcgmInjectFieldValue);

    /**
     * DEPRECATED
     * 
     * This method retrieves whether or not the policy manager thread loop is running
     * 0 = policy manager loop not running
     * !0 = policy manager loop running
     *
     * @param running 
     */
    void *deprecated;

    /**
     * This method retries the state of a field within the cache manager
     *
     * @param fieldInfo Structure to populate. fieldInfo->gpuId and fieldInfo->fieldId must
     *                  be populated on calling for this call to work
     *
     */
    dcgmReturn_t (*fpdcgmGetCacheManagerFieldInfo)(dcgmHandle_t pLwcmHandle, dcgmCacheManagerFieldInfo_t *fieldInfo);

    /**
     * Create fake entities for injection testing
     *
     * @param createFakeEntities Details about the number and type of entities to create
     *
     */
    dcgmReturn_t (*fpdcgmCreateFakeEntities)(dcgmHandle_t pDcgmHandle, dcgmCreateFakeEntities_t *createFakeEntities);

    /**
     * This method injects a sample into the cache manager
     *
     * @param entityGroupId
     * @param entityId
     * @param dcgmInjectFieldValue
     */
    dcgmReturn_t (*fpdcgmEntityInjectFieldValue)(dcgmHandle_t pLwcmHandle, 
                                                 dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId, 
                                                 dcgmInjectFieldValue_t *dcgmInjectFieldValue);
    
    /**
     * This method sets the link state of an entity's LwLink
     * 
     * dcgmHandle_t dcgmHandle
     * linkState    contains details about the link state to set
     */
    dcgmReturn_t (*fpdcgmSetEntityLwLinkLinkState)(dcgmHandle_t dcgmHandle, 
                                                   dcgmSetLwLinkLinkState_v1 *linkState);

}etblDCGMEngineTestInternal;



#ifdef	__cplusplus
}
#endif

#endif	/* DCGM_AGENT_INTERNAL_H */
