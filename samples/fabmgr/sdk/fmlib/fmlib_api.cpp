/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>

#include "fm_log.h"
#include "lw_fm_agent.h"
#include "fmLibClientConnHandler.h"
#include "FMErrorCodesInternal.h"
#include "fmlib_api.h"

typedef struct {
    // set to true when fmLibInit() is called. fmLibShutdown() will set this back to false
    bool isInitialized;

    // indicate whether logging was initialized.
    bool loggingIsInitialized;

    // how many threads are lwrrently using the client handler? This should be
    // 0 unless the client application uses threads to queue requests
    unsigned int clientConnHandlerRefCount;

    // pointer to our client connection handler. This cannot be freed unless the above 
    // clientConnHandlerRefCount reaches 0. This allows the API to send multiple requests
    // at a time (Note: Lwrrently this may not be required or used)
    fmLibClientConnHandler* clientConnHandler;

} fmLibGlobalCtxInfo_t;

static fmLibGlobalCtxInfo_t gFmLibGlobalCtx = {0}; // make it static so we don't export it

//
// Spin lock to control access to gFmLibGlobalCtx. This is declared outside
// of gFmLibGlobalCtx so gFmLibGlobalCtx can be memset to 0 in fmLibShutdown()
//
static volatile unsigned int gFmLibGlobalCtxLock = 0;

// logging-related elwironmental variables
#define FMLIB_ELW_LOG_LEVEL        "__FMLIB_LOG_LEVEL"
#define FMLIB_ELW_LOG_FILE_NAME    "__FMLIB_LOG_FILE_NAME"

/****************************************************************************************/
/*         All the helper functions. Not Public/exported from FMLib                     */
/****************************************************************************************/

static void
initializeFMLibLogging(void)
{
    //
    // the following logging option values are hardcoded
    // if needed we can expose them through elw variables.
    //    
    unsigned int appendToLogFile = 0; // over write the log
    unsigned int maxLogFileSize = 1; // 1GB size
    unsigned int useSysLog = 0; // use file based logging

    // the following options are read from evn variables
    unsigned int logLevel = 5; // default is at INFO
    char logFileName[FM_MAX_STR_LENGTH] = {0};
    char tempBuff[FM_MAX_STR_LENGTH] = {0};
    bool bInitLog = false;

    // get logging level
    if (lwosGetElw(FMLIB_ELW_LOG_LEVEL, tempBuff, 16) == 0) {
        logLevel = atoi(tempBuff);
    }

    // get log file name
    if (lwosGetElw(FMLIB_ELW_LOG_FILE_NAME, tempBuff, 16) == 0) {
        strncpy(logFileName, tempBuff, sizeof(logFileName));
        // init the logging based on whether a log file is specified or not
        // otherwise no logging
        bInitLog = true;
    }

    // set logging options/config
    if (bInitLog) {
        fabricManagerInitLog(logLevel, logFileName, appendToLogFile, maxLogFileSize, useSysLog);
        // remember the state so that we can do corresponding logging shutdown
        gFmLibGlobalCtx.loggingIsInitialized = true;
    }
}

static fmReturn_t
internalErrorCodeToPublicErrorCode(FMIntReturn_t intErrCode)
{
    switch (intErrCode) {
        case FM_INT_ST_OK:
            return FM_ST_SUCCESS;
        case FM_INT_ST_BADPARAM:
            return FM_ST_BADPARAM;
        case FM_INT_ST_NOT_SUPPORTED:
            return FM_ST_NOT_SUPPORTED;
        case FM_INT_ST_UNINITIALIZED:
            return FM_ST_UNINITIALIZED;
        case FM_INT_ST_CFG_TIMEOUT:
            return FM_ST_TIMEOUT;
        case FM_INT_ST_IN_USE:
            return FM_ST_IN_USE;
        case FM_INT_ST_NOT_CONFIGURED:
            return FM_ST_NOT_CONFIGURED;
        case FM_INT_ST_CONNECTION_NOT_VALID:
            return FM_ST_CONNECTION_NOT_VALID;
        case FM_INT_ST_LWLINK_ERROR:
            return FM_ST_LWLINK_ERROR;
        case FM_INT_ST_VERSION_MISMATCH:
            return FM_ST_VERSION_MISMATCH;
        default:
            return FM_ST_GENERIC_ERROR;
    }

    // default to keep compiler happy
    return FM_ST_GENERIC_ERROR;
}

static fmLibClientConnHandler*
acquireClientConnHandler()
{
    fmLibClientConnHandler *retVal = NULL;
    lwmlSpinLock(&gFmLibGlobalCtxLock);
    retVal = gFmLibGlobalCtx.clientConnHandler;
    gFmLibGlobalCtx.clientConnHandlerRefCount++;
    lwmlUnlock(&gFmLibGlobalCtxLock);
    return retVal;
}
static void
releaseClientConnHandler()
{   
    lwmlSpinLock(&gFmLibGlobalCtxLock);
    gFmLibGlobalCtx.clientConnHandlerRefCount--;
    lwmlUnlock(&gFmLibGlobalCtxLock);
}

static fmReturn_t
sendApiCommandToFMInstance(fmHandle_t pFmHandle, fmlib::CmdType cmdtype, void *apiCommandInfo,
                           unsigned int length, unsigned int version=0, FmRequest *request=0, 
                           unsigned int timeout=70000)
{
    fmlib::Msg *mpSendMsg;
    fmlib::Msg *mpRecvMsg;
    fmlib::Command *pCmdTemp;             /* Pointer to proto command for intermediate usage */
    fmlib::Command *recvCmd;              /* Pointer to proto command. Used as output parameter */
    fmReturn_t ret;
    FMIntReturn_t fmResult;
    fmlib::CmdArg *cmdArg;

    if(!apiCommandInfo)
        return FM_ST_BADPARAM;

    mpSendMsg = new fmlib::Msg;
    mpRecvMsg = new fmlib::Msg;

    pCmdTemp = new fmlib::Command;

    /* Add Command to the protobuf message */
    pCmdTemp->set_cmdtype(cmdtype);

    // add version to command
    pCmdTemp->set_version(version);

    cmdArg = new fmlib::CmdArg;

    cmdArg->set_blob(apiCommandInfo, length);
    pCmdTemp->set_allocated_arg(cmdArg);

    mpSendMsg->set_allocated_cmd(pCmdTemp);

    fmLibClientConnHandler* clientConnHandler = acquireClientConnHandler();

    /* Ilwoke method on the client side */
    fmResult = clientConnHandler->exchangeMsgBlocking(pFmHandle, mpSendMsg, mpRecvMsg, &recvCmd, timeout);

    releaseClientConnHandler();

    if (FM_INT_ST_OK != fmResult) {
        delete mpSendMsg;
        delete mpRecvMsg;
        return internalErrorCodeToPublicErrorCode(fmResult);
    }    
    
    if(!recvCmd->has_arg()) {
        delete mpSendMsg;
        delete mpRecvMsg;
        return FM_ST_GENERIC_ERROR;
    }
    
    if(!recvCmd->arg().has_blob()) {
        delete mpSendMsg;
        delete mpRecvMsg;
        return FM_ST_GENERIC_ERROR;
    }
    
    memcpy(apiCommandInfo, (void *)recvCmd->arg().blob().c_str(),
           recvCmd->arg().blob().size());

    // colwert internal FM command/api errors to public error codes
    ret = internalErrorCodeToPublicErrorCode((FMIntReturn_t)recvCmd->status());

    delete mpSendMsg;
    delete mpRecvMsg;
    return ret;
}

/****************************************************************************************/
/*          Public/Exported APIs from FMLib                                             */
/****************************************************************************************/

fmReturn_t
fmLibInit(void)
{
    // get the lock before accessing any states.
    lwmlSpinLock(&gFmLibGlobalCtxLock);

    // check whether the lib layer is already in initialized state
    if (gFmLibGlobalCtx.isInitialized) {
        // lib layer is already in initialized state
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_IN_USE;
    }

    // globals are uninitialized. zero out them
    memset(&gFmLibGlobalCtx, 0, sizeof(fmLibGlobalCtxInfo_t));

    //
    // do our logging init. this logging will be from the client process
    // which linked FMLib and not from FM process context and the log files
    // will be different. This logging is to troubleshoot anything happens
    // in the FMLib context.
    //

    initializeFMLibLogging();

    gFmLibGlobalCtx.clientConnHandlerRefCount = 0;

    //
    // create our client connection handler object which abstract socket connection to
    // running FM instance.
    //
    try {    
        gFmLibGlobalCtx.clientConnHandler = new fmLibClientConnHandler();
    } catch (std::runtime_error &e) {
        //
        // catch the error, so that we can return a value to the caller/client application.
        // the client app is expected to do the clean-up and exit. So not doing any explicit
        // clean-up
        //
        FM_LOG_ERROR("failed to allocate client socket connection object for FM communication");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_GENERIC_ERROR;
    }

    // fully initialized. mark accordingly
    gFmLibGlobalCtx.isInitialized = true;

    lwmlUnlock(&gFmLibGlobalCtxLock);

    FM_LOG_DEBUG("fmLibInit was successful");
    return FM_ST_SUCCESS;
}

fmReturn_t
fmLibShutdown(void)
{
    // get the lock before accessing any states.
    lwmlSpinLock(&gFmLibGlobalCtxLock);

    // check whether the lib layer was initialized for clean-up
    if (!gFmLibGlobalCtx.isInitialized) {
        // shutdown called without initialization
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    while (gFmLibGlobalCtx.clientConnHandlerRefCount > 0) {
        FM_LOG_INFO("Waiting to release reference count to client connection handler. Current RefCount: %d", 
                    gFmLibGlobalCtx.clientConnHandlerRefCount);
        // before sleep, release the lock so that other callers can decrement the ref count
        lwmlUnlock(&gFmLibGlobalCtxLock);
        sleep(1);
        // take the lock back and check
        lwmlSpinLock(&gFmLibGlobalCtxLock);
    }

    // all the reference to client handler is gone. delete it
    delete gFmLibGlobalCtx.clientConnHandler;
    gFmLibGlobalCtx.clientConnHandler = NULL;

    if (gFmLibGlobalCtx.loggingIsInitialized) {
        fabricManagerShutdownLog();
        gFmLibGlobalCtx.loggingIsInitialized = false;
    }

    // fully de-initialized. mark accordingly
    gFmLibGlobalCtx.isInitialized = false;
    lwmlUnlock(&gFmLibGlobalCtxLock);

    return FM_ST_SUCCESS;
}

fmReturn_t
fmConnect(fmConnectParams_t* connectParams, fmHandle_t* pFmHandle)
{
    fmReturn_t fmReturn;

    FM_LOG_DEBUG("entering fmConnect");

    // validate all the input parameters
    if ((NULL == connectParams) || (NULL == pFmHandle)) {
        // invalid arguments
        FM_LOG_DEBUG("fmConnect with invalid arguments");        
        return FM_ST_BADPARAM;
    }

    // the timeout has to be at least 1 ms
    if (!connectParams->timeoutMs) {
        // invalid arguments
        FM_LOG_DEBUG("fmConnect with invalid connection timeout value");        
        return FM_ST_BADPARAM;
    }

    if (connectParams->version != fmConnectParams_version) {
        FM_LOG_ERROR("fmConnect version mismatch detected. passed version: %X, internal version: %X",
                     connectParams->version, fmConnectParams_version);
        return FM_ST_VERSION_MISMATCH;
    }

    //
    // allow only one socket connection to go through. 
    // since we are under the global lock, accessing client handler directly
    //
    lwmlSpinLock(&gFmLibGlobalCtxLock);

    // check the lib initialization state before proceeding.
    if (!gFmLibGlobalCtx.isInitialized) {
        // fmConnect called before initializing API interface library
        FM_LOG_DEBUG("fmConnect called before initializing API interface library");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    FMIntReturn_t intRetVal;
    fmLibClientConnHandler* clientHandler = gFmLibGlobalCtx.clientConnHandler;
    intRetVal = clientHandler->openConnToRunningFMInstance(connectParams->addressInfo, pFmHandle,
                                                           connectParams->timeoutMs,
                                                           connectParams->addressIsUnixSocket);
    if (FM_INT_ST_OK != intRetVal) {
        FM_LOG_ERROR("failed to open connection to running fabric manager instance");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return internalErrorCodeToPublicErrorCode(intRetVal);
    }

    FM_LOG_DEBUG("Connected to FM instance ip %s pFmHandle %p",
                 connectParams->addressInfo, *pFmHandle);

    lwmlUnlock(&gFmLibGlobalCtxLock);

    return FM_ST_SUCCESS;
}

fmReturn_t
fmDisconnect(fmHandle_t pFmHandle)
{
    // get the lock before accessing any states.
    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        // fmDisconnect called before initializing API interface library
        FM_LOG_DEBUG("fmDisconnect called before initializing API interface library");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    //
    // lock is held to serialize connection and disconnection to running FM instance 
    // since we are under the global lock, accessing client handler directly
    //

    FMIntReturn_t intRetVal;
    fmLibClientConnHandler* clientHandler = gFmLibGlobalCtx.clientConnHandler;
    intRetVal = clientHandler->closeConnToRunningFMInstance(pFmHandle);
    if (FM_INT_ST_OK != intRetVal) {
        FM_LOG_WARNING("failed to close connection to running fabric manager instance");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return internalErrorCodeToPublicErrorCode(intRetVal);
    }

    lwmlUnlock(&gFmLibGlobalCtxLock);

    FM_LOG_DEBUG("fmDisconnect closed connection with handle %p", pFmHandle);
    return FM_ST_SUCCESS;
}


fmReturn_t
fmGetSupportedFabricPartitions(fmHandle_t pFmHandle,
                               fmFabricPartitionList_t *pFmFabricPartition)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        FM_LOG_ERROR("fmGetSupportedFabricPartitions called before FM Lib was initialized");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }
    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmFabricPartition == NULL || pFmHandle == NULL) {
        FM_LOG_ERROR("fmGetSupportedFabricPartitions called with invalid arguments");
        return FM_ST_BADPARAM;
    }

    if (pFmFabricPartition->version != fmFabricPartitionList_version) {
        FM_LOG_ERROR("fmGetSupportedFabricPartitions version mismatch detected. passed version: %X, internal version: %X",
                     pFmFabricPartition->version, fmFabricPartitionList_version);
        return FM_ST_VERSION_MISMATCH;        
    }

    //
    // we are not explicitly getting the caller buffer size. However, if the caller structure size changes,
    // then our above version check should fail. The internal fmFabricPartitionList_version value 
    // will change when fmFabricPartitionList_t size changes.
    //

    //
    // since fmFabricPartitionList_t size is very large, avoid using it as a stack variable and
    // instead allocate dynamic memory.
    //
    fm_msg_get_fabric_partition_t* pGetPartitionMsg = NULL;
    pGetPartitionMsg = (fm_msg_get_fabric_partition_t*) calloc(1, sizeof(fm_msg_get_fabric_partition_t));
    if (pGetPartitionMsg == NULL) {
        FM_LOG_ERROR("fmGetSupportedFabricPartitions failed to allocate required memory to hold partition list");
        return FM_ST_GENERIC_ERROR;
    }

    pGetPartitionMsg->version = fm_msg_get_fabric_partition_version;
    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_GET_SUPPORTED_PARTITIONS, (void*)pGetPartitionMsg, 
                                          sizeof(fm_msg_get_fabric_partition_t));

    if (fmResult == FM_ST_SUCCESS) {
         memcpy(pFmFabricPartition, &pGetPartitionMsg->fmFabricPartitionList,
                sizeof(pGetPartitionMsg->fmFabricPartitionList));
    }

    // free our locally allocated message after copying the partition information
    free(pGetPartitionMsg);

    return fmResult;
}

fmReturn_t
fmActivateFabricPartition(fmHandle_t pFmHandle,
                          fmFabricPartitionId_t partitionId)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmLibGlobalCtxLock);
        FM_LOG_ERROR("fmActivateFabricPartition called before FM Lib was initialized");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmHandle == NULL) {
        FM_LOG_ERROR("fmActivateFabricPartition called with invalid argument");
        return FM_ST_BADPARAM;
    }

    fm_msg_activate_fabric_partition_t activatePartitionMsg;
    activatePartitionMsg.partitionId = partitionId;
    activatePartitionMsg.version = fm_msg_activate_fabric_partition_version;

    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_ACTIVATE_PARTITION, (void*)&activatePartitionMsg, sizeof(activatePartitionMsg));

    return fmResult;    
}

fmReturn_t
fmActivateFabricPartitionWithVFs(fmHandle_t pFmHandle, fmFabricPartitionId_t partitionId, fmPciDevice_t *vfList, unsigned int numVfs)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmLibGlobalCtxLock);
        FM_LOG_ERROR("fmActivateFabricPartitionWithVFs called before FM Lib was initialized");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmLibGlobalCtxLock);

    if ((pFmHandle == NULL) || (vfList == NULL) || (numVfs == 0)) {
        FM_LOG_ERROR("fmActivateFabricPartitionWithVFs called with invalid argument");
        return FM_ST_BADPARAM;
    }

    fm_msg_activate_fabric_partition_vfs_t activatePartitionMsg;
    activatePartitionMsg.version = fm_msg_activate_fabric_partition_vfs_version;
    activatePartitionMsg.partitionId = partitionId;
    activatePartitionMsg.numVfs = std::min(FM_MAX_NUM_GPUS, (int)numVfs);
    memcpy(activatePartitionMsg.vfList, vfList, activatePartitionMsg.numVfs * sizeof(fmPciDevice_t));

    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_ACTIVATE_PARTITION_WITH_VFS, (void*)&activatePartitionMsg, sizeof(activatePartitionMsg));

    return fmResult;    
}

fmReturn_t
fmDeactivateFabricPartition(fmHandle_t pFmHandle,
                            fmFabricPartitionId_t partitionId)
{    
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);
    if (!gFmLibGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmLibGlobalCtxLock);
        FM_LOG_ERROR("fmDeactivateFabricPartition called before FM Lib was initialized");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmHandle == NULL) {
        FM_LOG_ERROR("fmDeactivateFabricPartition called with invalid argument");
        return FM_ST_BADPARAM;
    }

    fm_msg_deactivate_fabric_partition_t deactivatePartitionMsg;
    deactivatePartitionMsg.partitionId = partitionId;
    deactivatePartitionMsg.version = fm_msg_deactivate_fabric_partition_version;

    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_DEACTIVATE_PARTITION, (void*)&deactivatePartitionMsg, sizeof(deactivatePartitionMsg));

    return fmResult;   
}

fmReturn_t
fmSetActivatedFabricPartitions(fmHandle_t pFmHandle,
                               fmActivatedFabricPartitionList_t *pFmActivatedPartitionList)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        FM_LOG_ERROR("fmSetActivatedFabricPartitions called before FM Lib was initialized");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }
    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmActivatedPartitionList == NULL || pFmHandle == NULL) {
        FM_LOG_ERROR("fmSetActivatedFabricPartitions called with invalid arguments");
        return FM_ST_BADPARAM;
    }

    if (pFmActivatedPartitionList->version != fmActivatedFabricPartitionList_version) {
        FM_LOG_ERROR("fmSetActivatedFabricPartitions version mismatch detected. passed version: %X, internal version: %X",
                     pFmActivatedPartitionList->version, fmActivatedFabricPartitionList_version);
        return FM_ST_VERSION_MISMATCH;
    }

    fm_msg_set_activated_fabric_partition_list_t setActivatedPartitionMsg;
    setActivatedPartitionMsg.version = fm_msg_set_activated_fabric_partition_version;
    setActivatedPartitionMsg.fmActivatedFabricPartitionList.numPartitions = pFmActivatedPartitionList->numPartitions;

    for (uint i = 0; i < pFmActivatedPartitionList->numPartitions; i++) {
        setActivatedPartitionMsg.fmActivatedFabricPartitionList.partitionIds[i] = pFmActivatedPartitionList->partitionIds[i];
    }

    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_SET_ACTIVATED_PARTITION_LIST, (void*)&setActivatedPartitionMsg,
                                          sizeof(setActivatedPartitionMsg));

    return fmResult;
}

fmReturn_t
fmGetLwlinkFailedDevices(fmHandle_t pFmHandle, fmLwlinkFailedDevices_t *pFmLwlinkFailedDevices)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        FM_LOG_ERROR("fmGetLwlinkFailedDevices called before FM Lib was initialized");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }
    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmLwlinkFailedDevices == NULL || pFmHandle == NULL) {
        FM_LOG_ERROR("fmGetLwlinkFailedDevices called with invalid arguments");
        return FM_ST_BADPARAM;
    }

    if (pFmLwlinkFailedDevices->version != fmLwlinkFailedDevices_version) {
        FM_LOG_ERROR("fmLwlinkFailedDevices version mismatch detected. passed version: %X, internal version: %X",
                     pFmLwlinkFailedDevices->version, fmLwlinkFailedDevices_version);
        return FM_ST_VERSION_MISMATCH;
    }

    //
    // we are not explicitly getting the caller buffer size. However, if the caller structure size changes,
    // then our above version check should fail. The internal fmLwlinkFailedDevices_version value
    // will change when fmLwlinkFailedDevices_version size changes.
    //

    //
    // since fmLwlinkFailedDevices_version size is very large, avoid using it as a stack variable and
    // instead allocate dynamic memory.
    //
    fm_msg_get_lwlink_failed_devices_t* pGetLwlinkFailedDevicesMsg = NULL;
    pGetLwlinkFailedDevicesMsg = (fm_msg_get_lwlink_failed_devices_t*) calloc(1, sizeof(fm_msg_get_lwlink_failed_devices_t));
    if (pGetLwlinkFailedDevicesMsg == NULL) {
        FM_LOG_ERROR("fmGetLwlinkFailedDevices failed to allocate required memory to hold the lwlink failed device information");
        return FM_ST_GENERIC_ERROR;
    }

    pGetLwlinkFailedDevicesMsg->version = fm_msg_get_lwlink_failed_devices_version;
    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_GET_LWLINK_FAILED_DEVICES, (void*)pGetLwlinkFailedDevicesMsg,
                                          sizeof(fm_msg_get_lwlink_failed_devices_t));

    if (fmResult == FM_ST_SUCCESS) {
         memcpy(pFmLwlinkFailedDevices, &pGetLwlinkFailedDevicesMsg->fmLwlinkFailedDevices,
                sizeof(pGetLwlinkFailedDevicesMsg->fmLwlinkFailedDevices));
    }

    // free our locally allocated message after copying the lwlink failed device information
    free(pGetLwlinkFailedDevicesMsg);

    return fmResult;
}

fmReturn_t
fmGetUnsupportedFabricPartitions(fmHandle_t pFmHandle,
                                 fmUnsupportedFabricPartitionList_t *pFmUnsupportedFabricPartition)
{
    fmReturn_t fmResult;

    lwmlSpinLock(&gFmLibGlobalCtxLock);

    if (!gFmLibGlobalCtx.isInitialized) {
        FM_LOG_ERROR("fmGetUnsupportedSupportedFabricPartitions called before FM Lib was initialized");
        lwmlUnlock(&gFmLibGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }
    lwmlUnlock(&gFmLibGlobalCtxLock);

    if (pFmUnsupportedFabricPartition == NULL || pFmHandle == NULL) {
        FM_LOG_ERROR("fmGetUnsupportedSupportedFabricPartitions called with invalid arguments");
        return FM_ST_BADPARAM;
    }

    if (pFmUnsupportedFabricPartition->version != fmUnsupportedFabricPartitionList_version) {
        FM_LOG_ERROR("fmGetUnsupportedSupportedFabricPartitions version mismatch detected. passed version: %X, internal version: %X",
                     pFmUnsupportedFabricPartition->version, fmUnsupportedFabricPartitionList_version);
        return FM_ST_VERSION_MISMATCH;
    }

    //
    // we are not explicitly getting the caller buffer size. However, if the caller structure size changes,
    // then our above version check should fail. The internal fmUnsupportedFabricPartitionList_version value
    // will change when fmUnsupportedFabricPartitionList_t size changes.
    //

    //
    // since fmUnsupportedFabricPartitionList_t size is very large, avoid using it as a stack variable and
    // instead allocate dynamic memory.
    //
    fm_msg_get_unsupported_fabric_partition_t* pGetUnsupportedPartitionMsg = NULL;
    pGetUnsupportedPartitionMsg = (fm_msg_get_unsupported_fabric_partition_t*)
                                   calloc(1, sizeof(fm_msg_get_unsupported_fabric_partition_t));
    if (pGetUnsupportedPartitionMsg == NULL) {
        FM_LOG_ERROR("fmGetUnsupportedSupportedFabricPartitions failed to allocate required memory to hold partition list");
        return FM_ST_GENERIC_ERROR;
    }

    pGetUnsupportedPartitionMsg->version = fm_msg_get_unsupported_fabric_partition_version;
    fmResult = sendApiCommandToFMInstance(pFmHandle, fmlib::FM_GET_UNSUPPORTED_PARTITIONS, (void*)pGetUnsupportedPartitionMsg,
                                          sizeof(fm_msg_get_unsupported_fabric_partition_t));

    if (fmResult == FM_ST_SUCCESS) {
         memcpy(pFmUnsupportedFabricPartition, &pGetUnsupportedPartitionMsg->fmUnsupportedFabricPartitionList,
                sizeof(pGetUnsupportedPartitionMsg->fmUnsupportedFabricPartitionList));
    }

    // free our locally allocated message after copying the partition information
    free(pGetUnsupportedPartitionMsg);

    return fmResult;
}
