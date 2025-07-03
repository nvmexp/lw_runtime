/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
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
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <syslog.h>

#include "lwml_common.h"
#include "lwdcommon.h"
#include "prbdec.h"
#include "prblib.h"
#include "fmInternalApiConnHandler.h"
#include "FMErrorCodesInternal.h"
#include "g_fmInternalLib_pb.h"
#include "fm_internal_api_msg.h"
#include "fm_internal_api.h"

typedef struct {
    // set to true when fmInternalInit() is called. fmInternalShutdown() will set this back to false
    bool isInitialized;

} fmInternalGlobalCtxInfo_t;

static fmInternalGlobalCtxInfo_t gFmClientGlobalCtx = {0}; // make it static so we don't export it

//
// Spin lock to control access to gFmClientGlobalCtx. This is declared outside
// of gFmClientGlobalCtx so gFmClientGlobalCtx can be memset to 0 in fmInternalShutdown()
//
static volatile unsigned int gFmClientGlobalCtxLock = 0;

// buffer size used to encode protobuf messages
#define FMCLIENT_PRB_BUF_SIZE  512

/****************************************************************************************/
/*         All the helper functions. Not Public/exported from FMClient                     */
/****************************************************************************************/

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
        default:
            return FM_ST_GENERIC_ERROR;
    }

    // default to keep compiler happy
    return FM_ST_GENERIC_ERROR;
}

static fmReturn_t
sendApiCommandToFMInstance(uint32_t cmdType, void *apiCommandInfo,
                           unsigned int length, unsigned int version)
{
    fmReturn_t ret;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    char prbEncodeBuf[FMCLIENT_PRB_BUF_SIZE];

    if(!apiCommandInfo) {
        return FM_ST_BADPARAM;
    }

    PRB_ENCODER fmlibEncodeMsg;
    PRB_ENCODER *pSendMsg = &fmlibEncodeMsg;
    PRB_STATUS prbStatus;

    PRB_MSG fmlibDecodeMsg;
    PRB_MSG *pRecvMsg = &fmlibDecodeMsg;

    switch (cmdType) {
    case FMINTERNALLIB_FM_PREPARE_GPU_FOR_RESET:
    case FMINTERNALLIB_FM_SHUTDOWN_GPU_LWLINK:
    case FMINTERNALLIB_FM_RESET_GPU_LWLINK:
    case FMINTERNALLIB_FM_COMPLETE_GPU_RESET:
        break;

    default:
        syslog(LOG_ERR, "unknown Fabric Manager command type %d\n", cmdType);
        fprintf(stderr, "unknown Fabric Manager command type %d\n", cmdType);
        return FM_ST_GENERIC_ERROR;
    }

    // Encode the send message
    prbEncStart(pSendMsg, FMINTERNALLIB_MSG, (void *)prbEncodeBuf, FMCLIENT_PRB_BUF_SIZE);

    prbStatus = prbEncNestedStart(pSendMsg, FMINTERNALLIB_MSG_CMD);
    if (prbStatus != PRB_OK) {
        syslog(LOG_ERR, "failed to encode Fabric Manager command message %d with error %d\n",
               cmdType, prbStatus);
        fprintf(stderr, "failed to encode Fabric Manager command message %d with error %d\n",
                cmdType, prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    prbStatus = prbEncAddUInt32(pSendMsg, FMINTERNALLIB_COMMAND_CMDTYPE, cmdType);
    if (prbStatus != PRB_OK) {
        syslog(LOG_ERR, "request to encode Fabric Manager command type %d failed with error %d\n",
               cmdType, prbStatus);
        fprintf(stderr, "request to encode Fabric Manager command type %d failed with error %d\n",
                cmdType, prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    prbStatus = prbEncAddUInt32(pSendMsg, FMINTERNALLIB_COMMAND_VERSION, version);
    if (prbStatus != PRB_OK) {
        syslog(LOG_ERR, "failed to encode Fabric Manager command type %d version %d with error %d\n",
               cmdType, version, prbStatus);
        fprintf(stderr, "failed to encode Fabric Manager command type %d version %d with error %d\n",
                cmdType, version, prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    prbStatus = prbEncNestedStart(pSendMsg, FMINTERNALLIB_COMMAND_ARG);
    if (prbStatus != PRB_OK) {
        syslog(LOG_ERR, "failed to encode Fabric Manager command type %d arg with error %d\n",
               cmdType, prbStatus);
        fprintf(stderr, "failed to encode Fabric Manager command type %d arg with error %d\n",
                cmdType, prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    prbStatus = prbEncAddBytes(pSendMsg, FMINTERNALLIB_CMDARG_BLOB, (LwU8 *)apiCommandInfo, length);
    if ( prbStatus != PRB_OK) {
        syslog(LOG_ERR, "failed to encode Fabric Manager command type %d arg content with error %d\n",
               cmdType, prbStatus);
        fprintf(stderr, "failed to encode Fabric Manager command type %d arg content with error %d\n",
                cmdType, prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    prbEncNestedEnd(pSendMsg);
    prbEncNestedEnd(pSendMsg);

    // create receive message
    prbStatus = prbCreateMsg(pRecvMsg, FMINTERNALLIB_MSG);
    if (prbStatus != PRB_OK) {
        syslog(LOG_ERR, "failed to create Fabric Manager response message with error %d\n",
               prbStatus);
        fprintf(stderr, "failed to create Fabric Manager response message with error %d\n",
                prbStatus);
        return FM_ST_GENERIC_ERROR;
    }

    fmResult = exchangeMsgBlocking(pSendMsg, pRecvMsg);
    if (FM_INT_ST_OK != fmResult) {
        syslog(LOG_ERR, "failed to exchange message with running Fabric Manager instance with error %d\n",
                fmResult);
        fprintf(stderr, "failed to exchange message with running Fabric Manager instance with error %d\n",
                fmResult);
        prbDestroyMsg(pRecvMsg);
        return internalErrorCodeToPublicErrorCode(fmResult);
    }

    /* get the command arg */
    const PRB_MSG *pCmdArg;
    pCmdArg = prbGetMsg(pRecvMsg, FMINTERNALLIB_CMDARG);
    if (pCmdArg == NULL) {
        syslog(LOG_ERR, "received Fabric Manager response does not have command arg\n");
        fprintf(stderr, "received Fabric Manager response does not have command arg\n");
        prbDestroyMsg(pRecvMsg);
        return FM_ST_GENERIC_ERROR;
    }

    const PRB_FIELD *pField = prbGetField(pCmdArg, FMINTERNALLIB_CMDARG_BLOB);
    if (!pField || !pField->values->bytes.data) {
        syslog(LOG_ERR, "received Fabric Manager response does not have empty arg\n");
        fprintf(stderr, "received Fabric Manager response does not have empty arg\n");
        prbDestroyMsg(pRecvMsg);
        return FM_ST_GENERIC_ERROR;
    }

    memcpy(apiCommandInfo, (void *)pField->values->bytes.data,
           pField->values->bytes.len);

    /* get the response status */
    const PRB_MSG *pRecvdCmd;
    pRecvdCmd = prbGetMsg(pRecvMsg, FMINTERNALLIB_COMMAND);
    if (pRecvdCmd == NULL) {
        syslog(LOG_ERR, "received Fabric Manager response does not have command response\n");
        fprintf(stderr, "received Fabric Manager response does not have command response\n");
        prbDestroyMsg(pRecvMsg);
        return FM_ST_GENERIC_ERROR;
    }

    pField = prbGetField(pRecvdCmd, FMINTERNALLIB_COMMAND_STATUS);
    if (!pField) {
        syslog(LOG_ERR, "received Fabric Manager response does not have command status\n");
        fprintf(stderr, "received Fabric Manager response does not have command status\n");
        prbDestroyMsg(pRecvMsg);
        return FM_ST_GENERIC_ERROR;
    }

    // colwert internal FM command/api errors to public error codes
    ret = internalErrorCodeToPublicErrorCode((FMIntReturn_t)pField->values->uint32);

    prbDestroyMsg(pRecvMsg);
    return ret;
}

/****************************************************************************************/
/*          Public/Exported APIs from FMClient                                          */
/****************************************************************************************/

fmReturn_t
fmInternalInit(void)
{
    // get the lock before accessing any states.
    lwmlSpinLock(&gFmClientGlobalCtxLock);

    // check whether the client layer is already in initialized state
    if (gFmClientGlobalCtx.isInitialized) {
        // client layer is already in initialized state
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_IN_USE;
    }

    // globals are uninitialized. zero out them
    memset(&gFmClientGlobalCtx, 0, sizeof(fmInternalGlobalCtxInfo_t));

    // init connection to FM instance
    fmInternalApiConnHandlerInit();

    // fully initialized. mark accordingly
    gFmClientGlobalCtx.isInitialized = true;

    lwmlUnlock(&gFmClientGlobalCtxLock);

    //FM_LOG_DEBUG("fmInternalInit was successful");
    return FM_ST_SUCCESS;
}

fmReturn_t
fmInternalShutdown(void)
{
    // get the lock before accessing any states.
    lwmlSpinLock(&gFmClientGlobalCtxLock);

    // check whether the lib layer was initialized for clean-up
    if (!gFmClientGlobalCtx.isInitialized) {
        // shutdown called without initialization
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    disconnectFromFMInstance();

    // fully de-initialized. mark accordingly
    gFmClientGlobalCtx.isInitialized = false;

    lwmlUnlock(&gFmClientGlobalCtxLock);

    return FM_ST_SUCCESS;
}

fmReturn_t
fmInternalConnect(fmHandle_t* pFmHandle, unsigned int connTimeoutMs, unsigned int msgTimeoutMs)
{
    fmReturn_t fmReturn;
    fmConnectParams_t connectParams;

    // validate all the input parameters
    if (NULL == pFmHandle) {
        // invalid arguments
        syslog(LOG_ERR, "fmInternalConnect with invalid argument.\n");
        fprintf(stderr, "fmInternalConnect with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    struct stat st;
    /* Check the existence of the socket file. */
    if (stat(FM_INTERNAL_API_SOCKET_PATH, &st) < 0) {
        //FM_LOG_ERROR("Fabric Manager instance is not running.");
        return FM_ST_CONNECTION_NOT_VALID;
    }

    lwmlSpinLock(&gFmClientGlobalCtxLock);

    // check the lib initialization state before proceeding.
    if (!gFmClientGlobalCtx.isInitialized) {
        // fmInternalConnect called before initializing API interface library
        syslog(LOG_ERR, "fmInternalConnect called before initializing API interface library.\n");
        fprintf(stderr, "fmInternalConnect called before initializing API interface library.\n");
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    // if it is already connected
    if (isConnectedToFMInstance() == true) {
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_SUCCESS;
    }

    // make the connection to FM instance
    if (connectToFMInstance(connTimeoutMs, msgTimeoutMs) != FM_INT_ST_OK) {
        lwmlUnlock(&gFmClientGlobalCtxLock);
        syslog(LOG_ERR, "failed to connect to Fabric Manager instance.\n");
        fprintf(stderr, "failed to connect to Fabric Manager instance.\n");
        return FM_ST_CONNECTION_NOT_VALID;
    }

    *pFmHandle = getConnectionHandle();
    lwmlUnlock(&gFmClientGlobalCtxLock);

    return FM_ST_SUCCESS;
}

fmReturn_t
fmInternalDisconnect(fmHandle_t pFmHandle)
{
    // validate the handle
    if (pFmHandle != getConnectionHandle()) {
        // invalid arguments
        syslog(LOG_ERR, "fmInternalDisconnect with invalid argument.\n");
        fprintf(stderr, "fmInternalDisconnect with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    // get the lock before accessing any states.
    lwmlSpinLock(&gFmClientGlobalCtxLock);

    if (!gFmClientGlobalCtx.isInitialized) {
        // fmInternalDisconnect called before initializing API interface library
        syslog(LOG_ERR, "fmInternalDisconnect called before initializing API interface library.\n");
        fprintf(stderr, "fmInternalDisconnect called before initializing API interface library.\n");
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }

    // disconnect from FM instance
    disconnectFromFMInstance();

    lwmlUnlock(&gFmClientGlobalCtxLock);
    return FM_ST_SUCCESS;
}


fmReturn_t
fmPrepareGpuForReset(fmHandle_t pFmHandle, char *gpuUuid)
{
    // validate the handle
    if (pFmHandle != getConnectionHandle()) {
        // invalid arguments
        syslog(LOG_ERR, "fmPrepareGpuForReset with invalid argument.\n");
        fprintf(stderr, "fmPrepareGpuForReset with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    lwmlSpinLock(&gFmClientGlobalCtxLock);

    if (!gFmClientGlobalCtx.isInitialized) {
        syslog(LOG_ERR, "fmPrepareGpuForReset called before initializing API interface library.\n");
        fprintf(stderr, "fmPrepareGpuForReset called before initializing API interface library.\n");
        lwmlUnlock(&gFmClientGlobalCtxLock);
        return FM_ST_UNINITIALIZED;
    }
    lwmlUnlock(&gFmClientGlobalCtxLock);

    if (gpuUuid == NULL || (isConnectedToFMInstance() == false)) {
        syslog(LOG_ERR, "fmPrepareGpuForReset with invalid argument.\n");
        fprintf(stderr, "fmPrepareGpuForReset with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    fm_msg_prepare_gpu_for_reset_t msg;
    msg.version = fm_msg_prepare_gpu_for_reset_version;
    strncpy(msg.gpuUuid, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    return sendApiCommandToFMInstance(FMINTERNALLIB_FM_PREPARE_GPU_FOR_RESET,
                                      (void*)&msg, sizeof(msg),
                                      fm_msg_prepare_gpu_for_reset_version);
}

fmReturn_t
fmShutdownGpuLWLinks(fmHandle_t pFmHandle, char *gpuUuid)
{
    // validate the handle
    if (pFmHandle != getConnectionHandle()) {
        // invalid arguments
        syslog(LOG_ERR, "fmShutdownGpuLWLinks with invalid argument.\n");
        fprintf(stderr, "fmShutdownGpuLWLinks with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    lwmlSpinLock(&gFmClientGlobalCtxLock);

    if (!gFmClientGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmClientGlobalCtxLock);
        syslog(LOG_ERR, "fmShutdownGpuLWLinks called before initializing API interface library.\n");
        fprintf(stderr, "fmShutdownGpuLWLinks called before initializing API interface library.\n");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmClientGlobalCtxLock);

    if (gpuUuid == NULL || (isConnectedToFMInstance() == false)) {
        syslog(LOG_ERR, "fmShutdownGpuLWLinks with invalid argument.\n");
        fprintf(stderr, "fmShutdownGpuLWLinks with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    fm_msg_shutdown_gpu_lwlink_t msg;
    msg.version = fm_msg_shutdown_gpu_lwlink_version;
    strncpy(msg.gpuUuid, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    return sendApiCommandToFMInstance(FMINTERNALLIB_FM_SHUTDOWN_GPU_LWLINK,
                                      (void*)&msg, sizeof(msg),
                                      fm_msg_shutdown_gpu_lwlink_version);
}

fmReturn_t
fmResetGpuLWLinks(fmHandle_t pFmHandle, char *gpuUuid)
{
    // validate the handle
    if (pFmHandle != getConnectionHandle()) {
        // invalid arguments
        syslog(LOG_ERR, "fmResetGpuLWLinks with invalid argument.\n");
        fprintf(stderr, "fmResetGpuLWLinks with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    lwmlSpinLock(&gFmClientGlobalCtxLock);

    if (!gFmClientGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmClientGlobalCtxLock);
        syslog(LOG_ERR, "fmResetGpuLWLinks called before initializing API interface library.\n");
        fprintf(stderr, "fmResetGpuLWLinks called before initializing API interface library.\n");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmClientGlobalCtxLock);

    if (gpuUuid == NULL || (isConnectedToFMInstance() == false)) {
        syslog(LOG_ERR, "fmResetGpuLWLinks with invalid argument.\n");
        fprintf(stderr, "fmResetGpuLWLinks with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    fm_msg_reset_gpu_lwlink_t msg;
    msg.version = fm_msg_reset_gpu_lwlink_version;
    strncpy(msg.gpuUuid, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    return sendApiCommandToFMInstance(FMINTERNALLIB_FM_RESET_GPU_LWLINK,
                                      (void*)&msg, sizeof(msg),
                                      fm_msg_reset_gpu_lwlink_version);
}

fmReturn_t
fmCompleteGpuReset(fmHandle_t pFmHandle, char *gpuUuid)
{    
    // validate the handle
    if (pFmHandle != getConnectionHandle()) {
        // invalid arguments
        syslog(LOG_ERR, "fmCompleteGpuReset with invalid argument.\n");
        fprintf(stderr, "fmCompleteGpuReset with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    lwmlSpinLock(&gFmClientGlobalCtxLock);

    if (!gFmClientGlobalCtx.isInitialized) {
        lwmlUnlock(&gFmClientGlobalCtxLock);
        syslog(LOG_ERR, "fmCompleteGpuReset called before initializing API interface library.\n");
        fprintf(stderr, "fmCompleteGpuReset called before initializing API interface library.\n");
        return FM_ST_UNINITIALIZED;
    }

    lwmlUnlock(&gFmClientGlobalCtxLock);

    if (gpuUuid == NULL || (isConnectedToFMInstance() == false)) {
        syslog(LOG_ERR, "fmCompleteGpuReset with invalid argument.\n");
        fprintf(stderr, "fmCompleteGpuReset with invalid argument.\n");
        return FM_ST_BADPARAM;
    }

    fm_msg_complete_gpu_reset_t msg;
    msg.version = fm_msg_complete_gpu_reset_version;
    strncpy(msg.gpuUuid, gpuUuid, FM_UUID_BUFFER_SIZE - 1);

    return sendApiCommandToFMInstance(FMINTERNALLIB_FM_COMPLETE_GPU_RESET,
                                      (void*)&msg, sizeof(msg),
                                      fm_msg_complete_gpu_reset_version);
}
