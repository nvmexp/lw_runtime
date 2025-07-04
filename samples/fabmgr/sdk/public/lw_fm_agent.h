
/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#ifndef LW_FM_AGENT_H
#define LW_FM_AGENT_H
#ifdef __cplusplus
extern "C" {
#endif
#include "lw_fm_types.h"
#define DECLDIR
/***************************************************************************************************/
/** @defgroup FMAPI_Admin Administrative
 *   
 *  This chapter describes the administration interfaces for Fabric Manager API interface library.
 *  It is the user's responsibility to call \ref fmLibInit() before calling any other methods, 
 *  and \ref fmLibShutdown() once Fabric Manager is no longer being used. The APIs in Administrative
 *  module can be broken down into following categories:
 *  @{
 */
/***************************************************************************************************/
/**
 * This method is used to initialize the Fabric Manager API interface library. This must be 
 * called before fmConnect()
 * 
 *  * @return 
 *        - \ref FM_ST_SUCCESS              if FM API interface library has been properly initialized
 *        - \ref FM_ST_IN_USE               FM API interface library is already in initialized state
 *        - \ref FM_ST_GENERIC_ERROR        A generic, unspecified error oclwrred
 */
fmReturn_t DECLDIR fmLibInit(void);
/**
 * This method is used to shut down the Fabric Manager API interface library. Any remote connections
 * established through fmConnect() will be shut down as well.
 *  
 * @return 
 *        - \ref FM_ST_SUCCESS           if FM API interface library has been properly shut down
 *        - \ref FM_ST_UNINITIALIZED     FM API interface library was not in initialized state
 */
fmReturn_t DECLDIR fmLibShutdown(void);
/**
 * This method is used to connect to a running instance of Fabric Manager. Fabric Manager instance is
 * started as part of system service or manually by the SysAdmin. This connection will be used by the
 * APIs to exchange information to the running Fabric Manager instance.
 *
 * @param connectParams IN : Valid IP address for the remote host engine to connect to.
 *                           If addressInfo is specified as x.x.x.x it will attempt to connect to the default
 *                           port specified by FM_CMD_PORT_NUMBER
 *                           If addressInfo is specified as x.x.x.x:yyyy it will attempt to connect to the
 *                           port specified by yyyy
 *                           To connect to an FM instance that was started with unix domain socket, 
 *                           fill the socket path in addressInfo member and set addressIsUnixSocket flag.
 *
 *                           For additional connection parameters. See \ref fmConnectParams_t for details.
 * @param pfmHandle     OUT : Fabric Manager API interface abstracted handle for subsequent API calls.
 *
 * @return
 *         - \ref FM_ST_SUCCESS                successfully connected to the FM instance
 *         - \ref FM_ST_CONNECTION_NOT_VALID   if the FM instance could not be reached
 *         - \ref FM_ST_UNINITIALIZED          if FM interface library has not been initialized with \ref fmLibInit.
 *         - \ref FM_ST_BADPARAM               if pFmHandle is NULL or provided IP Address/format is invalid
 *         - \ref FM_ST_VERSION_MISMATCH       if the expected and provided versions of connectParams do not match
 */
 fmReturn_t DECLDIR fmConnect(fmConnectParams_t *connectParams, fmHandle_t *pFmHandle);
/**
 * This method is used to disconnect from a Fabric Manager instance.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @return
 *         - \ref FM_ST_SUCCESS             if we successfully disconnected from the FM instance
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 */
fmReturn_t DECLDIR fmDisconnect(fmHandle_t pFmHandle);
/** @} */ // Closing for FMAPI_Admin
/******************************************************************************************************/
/** @defgroup FMAPI_FabricPartition Fabric Partition Related APIs
 *   
 *  This chapter describes the APIs for Fabric Partition management for Shared LWSwitch and vGPU Models 
 *  @{
 */
/*******************************************************************************************************/
/**
 * This method is used to query all the supported fabric partitions in an LWSwitch based system.
 * These fabric partitions allow users to assign specified GPUs to a guestOS as part of multitenancy
 * with necessary LWLink isolation.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param pFmFabricPartition  OUT: List of lwrrently supported fabric partition information.
 *
 * @return
 *         - \ref FM_ST_SUCCESS             successfully queried the list of supported partitions
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            Invalid input parameters
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_VERSION_MISMATCH    if the expected and provided versions of pFmFabricPartition do not match
 */
fmReturn_t DECLDIR fmGetSupportedFabricPartitions(fmHandle_t pFmHandle, fmFabricPartitionList_t *pFmFabricPartition);
/**
 * This method is used to activate an available fabric partition in an LWSwitch based system.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param partitionId  IN: The partition id to be activated
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified partition is activated successfully
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or unsupported partition id
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_IN_USE              specified partition is already active
 *         - \ref FM_ST_LWLINK_ERROR        LWLink error/training failure oclwrred when activating the partition
 */
fmReturn_t DECLDIR fmActivateFabricPartition(fmHandle_t pFmHandle, fmFabricPartitionId_t partitionId);
/**
 * This method is used to activate an available fabric partition with VFs in an LWSwitch based system.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param partitionId  IN: The partition id to be activated
 *
 * @param vfList IN: List of VFs associated with physical GPUs in the partition.
 *                   Please note that the order of VFs should be associated with actual physical GPUs in the partition.
 *
 * @param numVfs IN: Number of VFs
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified partition is activated successfully
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or unsupported partition id
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_IN_USE              specified partition is already active
 *         - \ref FM_ST_LWLINK_ERROR        LWLink error/training failure oclwrred when activating the partition
 */
fmReturn_t DECLDIR fmActivateFabricPartitionWithVFs(fmHandle_t pFmHandle, fmFabricPartitionId_t partitionId, fmPciDevice_t *vfList, unsigned int numVfs);
/**
 * This method is used to deactivate a previously activated fabric partition in an LWSwitch based system.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param partitionId  IN: The partition id to be deactivated
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified partition is deactivated successfully
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or unsupported partition id
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_UNINITIALIZED       specified partition is not activated
 *         - \ref FM_ST_LWLINK_ERROR        LWLink error/training failure oclwrred when deactivating the partition
 */
fmReturn_t DECLDIR fmDeactivateFabricPartition(fmHandle_t pFmHandle, fmFabricPartitionId_t partitionId);
/**
 * This method is used to set a list of lwrrently activated fabric partitions to Fabric Manager after its restart.
 * This call should be made with number of partitions as zero even if there is no active partitions
 * when Fabric Manager is restarted.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param pFmActivatedPartitionList  IN: List of lwrrently activated fabric partition.
 * @return
 *         - \ref FM_ST_SUCCESS             Fabric Manager state is updated with active partition information
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            A bad parameter was passed.
  *        - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       Requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager is initializing and no data available.
 *         - \ref FM_ST_VERSION_MISMATCH    if the expected and provided versions of pFmActivatedPartitionList do not match
 */
fmReturn_t DECLDIR fmSetActivatedFabricPartitions(fmHandle_t pFmHandle, fmActivatedFabricPartitionList_t *pFmActivatedPartitionList);
/**
 * This method is used to query all GPUs and LWSwitches with failed LWLinks as part of Fabric Manager initialization
 *
 * This API is not supported when Fabric Manager is running in Shared LWSwitch
 * multi-tenancy resiliency restart (--restart) mode
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param pFmLwlinkFailedDevices  OUT: List of GPU or LWSwitch devices that have failed LWLinks
 *
 * @return
 *         - \ref FM_ST_SUCCESS             successfully queried the list of devices with failed LWLinks
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            Invalid input parameters
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_VERSION_MISMATCH    if the expected and provided versions of pFmFabricPartition do not match
 */
fmReturn_t DECLDIR fmGetLwlinkFailedDevices(fmHandle_t pFmHandle, fmLwlinkFailedDevices_t *pFmLwlinkFailedDevices);
/***************************************************************************************************/
/**
 * This method is used to query all the unsupported fabric partitions when Fabric Manager is
 * running in Shared LWSwitch multi-tenancy mode.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param pFmFabricPartition  OUT: List of unsupported fabric partitions on the system.
 *
 * @return
 *         - \ref FM_ST_SUCCESS             successfully queried the list of unsupported partitions
 *         - \ref FM_ST_UNINITIALIZED       if FM interface library has not been initialized with \ref fmLibInit
 *         - \ref FM_ST_BADPARAM            Invalid input parameters
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is initializing and no data
 *         - \ref FM_ST_VERSION_MISMATCH    if the expected and provided versions of pFmFabricPartition do not match
 */
fmReturn_t DECLDIR fmGetUnsupportedFabricPartitions(fmHandle_t pFmHandle,
                                                    fmUnsupportedFabricPartitionList_t *pFmUnupportedFabricPartition);
/** @} */ // Closing for FMAPI_FabricPartition
#ifdef __cplusplus
}
#endif

#endif /* LW_FM_AGENT_H */

