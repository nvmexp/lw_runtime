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
 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
 
#include "fm_log.h"
#include "lw_fm_types.h"
#include "fabricmanager.pb.h"
#include "GlobalFMInternalCmdServer.h"
#include "topology.pb.h"
#include "fm_internal_api.h"
#include "fm_internal_api_msg.h"

GlobalFMInternalCmdServer::GlobalFMInternalCmdServer(GlobalFabricManager* gfm,
                                                     GlobalFmApiHandler *pApiHandler)
    : FmSocket( 0, FM_INTERNAL_API_SOCKET_PATH, false, FM_SERVER_WORKER_NUM_THREADS)
{
    mpGfm = gfm;
    mpApiHandler = pApiHandler;
};
 
GlobalFMInternalCmdServer::~GlobalFMInternalCmdServer()
{
    // stop our listening thread/server
    StopServer();
};
 
int
GlobalFMInternalCmdServer::OnRequest(fm_request_id_t requestId, FmServerConnection* pConnection)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    fmInternalLib::Command *pCmdWrapper;
    vector<fmInternalLib::Command *>::iterator cmdIterator;
    FmSocketMessage *pMessageRecvd;        /* Pointer to FM Socket Message */
    FmRequest *pFmServerRequest;           /* Pointer to FM Request */
    fmInternalLib::Msg *fmInternalLibProtoMsg;
    vector<fmInternalLib::Command *> vecCmds; /* To store reference to commands inside the protobuf message */
    bool isComplete;                 /* Flag to determine if the request handling is complete */
    int st;                          /* Status code */
 
    if (!pConnection) {
        FM_LOG_ERROR("socket connection used by Fabric Manager Internal API interface is not valid");
        FM_SYSLOG_ERR("socket connection used by Fabric Manager Internal API interface is not valid");
        return -1;
    }
    /* Add Reference to the connection as the copy is used in this message */
    pConnection->IncrReference();
 
    /**
     * Before processing the request check if the connection is still in 
     * active state. If the connection is not active then don't even proceed 
     * and mark the request as completed. The CompleteRequest will delete the
     * connection bindings if this request is the last entity holding on to the 
     * connection even when the connection is in inactive state.
     */
    if (!pConnection->IsConnectionActive()) {
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        FM_LOG_ERROR("socket connection used by Fabric Manager API interface is not in active state");
        return 0;
    }
 
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr(inet_ntoa(remoteAddr.sin_addr));
 
    pFmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pFmServerRequest) {
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        FM_LOG_ERROR("failed to get message/request from socket connection used by Fabric Manager API interface");
        return -1;
    }    
 
    if (pFmServerRequest->MessageCount() != 1) {
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        FM_LOG_ERROR("expected only single request message on socket connection used by Fabric Manager API interface");
        return -1;
    }
 
    // Get the message received corresponding to the request id
    pMessageRecvd = pFmServerRequest->GetNextMessage();
 
    if (NULL == pMessageRecvd) {
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        FM_LOG_DEBUG("failed to get message for request id %d.", (int)requestId);
        return -1;
    }

    fmInternalLibProtoMsg = new fmInternalLib::Msg;

    if (true != fmInternalLibProtoMsg->ParseFromArray((char *)pMessageRecvd->GetContent(), pMessageRecvd->GetLength())) {
        FM_LOG_ERROR("failed to parse and decode message received on socket connection used by Fabric Manager API interface");
        delete fmInternalLibProtoMsg;
        delete pMessageRecvd;
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }
 
    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pMessageRecvd;
    pMessageRecvd = NULL;
 
    fmInternalLib::Command  *pCmd = (fmInternalLib::Command*)&fmInternalLibProtoMsg->cmd();

    if (pCmd == NULL) {
        FM_LOG_ERROR("failed to parse Fabric Manager API command from the received request message\n");
        delete fmInternalLibProtoMsg;
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    processMessage(pCmd);
 
    //send FM Message back to the client
    st = sendReplyToClient(fmInternalLibProtoMsg, pConnection, requestId, FM_MSG_PROTO_RESPONSE);
    if( st != 0) {
        FM_LOG_ERROR("failed to send fabric manager API command response message back to the requested client");
    }

    delete fmInternalLibProtoMsg;
    pConnection->CompleteRequest(requestId);
    pConnection->DecrReference();
    return 0;
}
 
void
GlobalFMInternalCmdServer::processMessage(fmInternalLib::Command *pCmd)
{
    fmInternalLib::CmdType cmdtype = pCmd->cmdtype();
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    fmReturn_t ret;
    void *apiCommandInfo = (void*)pCmd->arg().blob().c_str();
 
    FM_LOG_DEBUG("processMessage received command %d\n", cmdtype);

    switch (cmdtype) {
        case fmInternalLib::FM_PREPARE_GPU_FOR_RESET:
            {
                fm_msg_prepare_gpu_for_reset_t *prepareGpuForReset = (fm_msg_prepare_gpu_for_reset_t*) apiCommandInfo;
                if (prepareGpuForReset->version != fm_msg_prepare_gpu_for_reset_version) {
                    FM_LOG_ERROR("FM Prepare GPU for Reset version mismatch detected. passed version: %X, internal version: %X",
                                 prepareGpuForReset->version, fm_msg_prepare_gpu_for_reset_version);
                    pCmd->set_status(FM_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->prepareGpuForReset(prepareGpuForReset->gpuUuid);
                break;
            }

        case fmInternalLib::FM_SHUTDOWN_GPU_LWLINK:
            {
                fm_msg_shutdown_gpu_lwlink_t *shutdownGpuLwlink = (fm_msg_shutdown_gpu_lwlink_t*) apiCommandInfo;
                if (shutdownGpuLwlink->version != fm_msg_shutdown_gpu_lwlink_version) {
                    FM_LOG_ERROR("FM Shutdown GPU LWLink version mismatch detected. passed version: %X, internal version: %X",
                            shutdownGpuLwlink->version, fm_msg_shutdown_gpu_lwlink_version);
                    pCmd->set_status(FM_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->shutdownGpuLwlink(shutdownGpuLwlink->gpuUuid);
                break;
            }
        case fmInternalLib::FM_RESET_GPU_LWLINK:
            {  
                fm_msg_reset_gpu_lwlink_t *resetGpuLwlink = (fm_msg_reset_gpu_lwlink_t*) apiCommandInfo;
                if (resetGpuLwlink->version != fm_msg_reset_gpu_lwlink_version) {
                    FM_LOG_ERROR("FM Reset GPU LWLink version mismatch detected. passed version: %X, internal version: %X",
                                 resetGpuLwlink->version, fm_msg_reset_gpu_lwlink_version);
                    pCmd->set_status(FM_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->resetGpuLwlink(resetGpuLwlink->gpuUuid);
                break;
            }
        case fmInternalLib::FM_COMPLETE_GPU_RESET:
            {
                fm_msg_complete_gpu_reset_t *completeGpuReset = (fm_msg_complete_gpu_reset_t*) apiCommandInfo;
                if (completeGpuReset->version != fm_msg_complete_gpu_reset_version) {
                    FM_LOG_ERROR("FM Complete GPU Reset version mismatch detected. passed version: %X, internal version: %X",
                            completeGpuReset->version, fm_msg_complete_gpu_reset_version);
                    pCmd->set_status(FM_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->completeGpuReset(completeGpuReset->gpuUuid);
                break;
            }
        default:
            break;        
    }
 
    pCmd->set_status(fmResult);
}
 
int 
GlobalFMInternalCmdServer::sendReplyToClient(fmInternalLib::Msg *fmInternalLibProtoMsg, FmServerConnection *pConnection,
                                             fm_request_id_t requestId, unsigned int msgType)
{
    char *msgToSend;
    unsigned int msgLen;    
    FmSocketMessage fmReply;
    int st;
 
    msgLen = fmInternalLibProtoMsg->ByteSize();
    msgToSend = new char [msgLen];
    fmInternalLibProtoMsg->SerializeToArray(msgToSend, msgLen);
 
    fmReply.UpdateMsgHdr(msgType, requestId, FM_PROTO_ST_SUCCESS, msgLen);
    fmReply.UpdateMsgContent(msgToSend, msgLen);
 
    st = pConnection->SetOutputBuffer(&fmReply);

    // free send messsage buffer
    delete [] msgToSend;

    if (st) {
        return st;
    }
 
    return 0;
}

