
/*
 * Copyright 1993-201 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#ifndef FM_INTERNAL_API_H
#define FM_INTERNAL_API_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include "lw_fm_types.h"

#define FM_INTERNAL_API_SOCKET_PATH "/var/run/lwpu-fabricmanager/socket"

#define DECLDIR
/***************************************************************************************************/
/** @defgroup FM_INTERNAL_API_Admin Administrative
 *   
 *  This chapter describes the administration interfaces for Fabric Manager internal API library.
 *  It is the user's responsibility to call \ref fmInternalInit() before calling any other methods,
 *  and \ref fmInternalShutdown() once Fabric Manager is no longer being used. The APIs in Administrative
 *  module can be broken down into following categories:
 *  @{
 */
/***************************************************************************************************/
/**
 * This method is used to initialize the Fabric Manager internal API library. This must be
 * called before fmInternalConnect()
 * 
 *  * @return 
 *        - \ref FM_ST_SUCCESS              if FM internal API library has been properly initialized
 *        - \ref FM_ST_IN_USE               FM internal API library is already in initialized state
 *        - \ref FM_ST_GENERIC_ERROR        A generic, unspecified error oclwrred
 */
fmReturn_t DECLDIR fmInternalInit(void);
/**
 * This method is used to shut down the Fabric Manager internal API library. Any remote connections
 * established through fmInternalConnect() will be shut down as well.
 *  
 * @return 
 *        - \ref FM_ST_SUCCESS           if FM internal API library has been properly shut down
 *        - \ref FM_ST_UNINITIALIZED     FM internal API library was not in initialized state
 */
fmReturn_t DECLDIR fmInternalShutdown(void);
/**
 * This method is used to connect to a running instance of Fabric Manager. Fabric Manager instance is
 * started as part of system service or manually by the SysAdmin. This connection will be used by the
 * APIs to exchange information to the running Fabric Manager instance.
 *
 * @param pfmHandle     OUT : Fabric Manager internal API interface abstracted handle for subsequent API calls.
 *
 * @return
 *         - \ref FM_ST_SUCCESS                successfully connected to the FM instance
 *         - \ref FM_ST_CONNECTION_NOT_VALID   if the FM instance could not be reached
 *         - \ref FM_ST_UNINITIALIZED          if FM internal API library has not been initialized with \ref fmInternalInit.
 *         - \ref FM_ST_BADPARAM               if pFmHandle is NULL or provided IP Address/format is invalid
 *         - \ref FM_ST_VERSION_MISMATCH       if the expected and provided versions of connectParams do not match
 *         - \ref FM_ST_CONNECTION_NOT_VALID   if fabric manager instance is not running
 */
 fmReturn_t DECLDIR fmInternalConnect(fmHandle_t *pFmHandle, unsigned int connTimeoutMs, unsigned int msgTimeoutMs);
/**
 * This method is used to disconnect from a Fabric Manager instance.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param connTimeoutMs IN: When attempting to connect to the running FM instance,
 *                          how long should we wait in milliseconds before giving up
 *
 * @param msgTimeoutMs IN: When attempting to exchange message with the running FM instance,
 *                         how long should we wait in milliseconds before giving up
 *
 * @return
 *         - \ref FM_ST_SUCCESS             if we successfully disconnected from the FM instance
 *         - \ref FM_ST_UNINITIALIZED       if FM internal API library has not been initialized with \ref fmInternalInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 */
fmReturn_t DECLDIR fmInternalDisconnect(fmHandle_t pFmHandle);
/** @} */ // Closing for FM_INTERNAL_API_Admin
/***************************************************************************************************/
/** @defgroup FM_INTERNAL_API_GpuReset GPU reset Related APIs
 *   
 *  This chapter describes the APIs for GPU reset
 *  @{
 */
/***************************************************************************************************/
/**
 * This method is used to prepare the specified GPU for reset.
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param gpuUuid  IN: UUID of the GPU to be reset
 *
 * @return
 *         - \ref FM_ST_SUCCESS             successfully prepared the GPU for reset.
 *         - \ref FM_ST_UNINITIALIZED       if FM internal API library has not been initialized with \ref fmInternalInit
 *         - \ref FM_ST_BADPARAM            Invalid input parameters
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is still initializing
 */
fmReturn_t DECLDIR fmPrepareGpuForReset(fmHandle_t pFmHandle, char *gpuUuid);
/**
 * This method is used to shut down the specified GPU LWLinks
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param gpuUuid  IN: UUID of the GPU to be reset
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified GPU LWLinks are shutdown successfully.
 *         - \ref FM_ST_UNINITIALIZED       if FM internal API library has not been initialized with \ref fmInternalInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or invalid gpuUuid
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is still initializing
 */
fmReturn_t DECLDIR fmShutdownGpuLWLinks(fmHandle_t pFmHandle, char *gpuUuid);
/**
 * This method is used to reset the specified GPU LWLinks
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param gpuUuid  IN: UUID of the GPU to be reset
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified GPU LWLinks are reset successfully.
 *         - \ref FM_ST_UNINITIALIZED       if FM internal API library has not been initialized with \ref fmInternalInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or invalid gpuUuid
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is still initializing
 */
fmReturn_t DECLDIR fmResetGpuLWLinks(fmHandle_t pFmHandle, char *gpuUuid);
/**
 * This method is used to complete the reset of the specified GPU
 *
 * @param pFmHandle IN:  Handle that came form \ref fmConnect
 *
 * @param gpuUuid  IN: UUID of the GPU that has completed reset
 *
 * @return
 *         - \ref FM_ST_SUCCESS             Specified GPU's reset is completed successfully.
 *         - \ref FM_ST_UNINITIALIZED       if FM internal API library has not been initialized with \ref fmInternalInit
 *         - \ref FM_ST_BADPARAM            if pFmHandle is not a valid handle or invalid gpuUuid
 *         - \ref FM_ST_GENERIC_ERROR       if an unspecified internal error oclwrred
 *         - \ref FM_ST_NOT_SUPPORTED       requested feature is not supported or enabled
 *         - \ref FM_ST_NOT_CONFIGURED      Fabric Manager instance is still initializing
 */
fmReturn_t DECLDIR fmCompleteGpuReset(fmHandle_t pFmHandle, char *gpuUuid);


/** @} */ // Closing for FM_INTERNAL_API_GpuReset
#ifdef __cplusplus
}
#endif

#endif /* FM_INTERNAL_API_H */

