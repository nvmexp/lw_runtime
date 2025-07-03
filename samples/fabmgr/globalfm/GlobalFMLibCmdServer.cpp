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
 
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
 
#include "fm_log.h"
#include "lw_fm_types.h"
#include "fabricmanager.pb.h"
#include "GlobalFMLibCmdServer.h"
#include "topology.pb.h"
#include "fmlib_api.h"
 
GlobalFMLibCmdServer::GlobalFMLibCmdServer(GlobalFabricManager* gfm,
                                           GlobalFmApiHandler *pApiHandler,
                                           int portNumber, char *sockpath, int isTCP )
    : FmSocket( portNumber, sockpath, isTCP, FM_SERVER_WORKER_NUM_THREADS)
{
    mpGfm = gfm;
    mpApiHandler = pApiHandler;
};
 
GlobalFMLibCmdServer::~GlobalFMLibCmdServer()
{
    // stop our listening thread/server
    StopServer();
};
 
int
GlobalFMLibCmdServer::OnRequest(fm_request_id_t requestId, FmServerConnection* pConnection)
{
    int ret;
    FmSocketMessage fmSendMsg;
    char *bufToSend;
    unsigned int msgLength;
    fmlib::Command *pCmdWrapper;
    vector<fmlib::Command *>::iterator cmdIterator;
    FmSocketMessage *pMessageRecvd;        /* Pointer to FM Socket Message */
    FmRequest *pFmServerRequest;           /* Pointer to FM Request */
    fmlib::Msg *fmlibProtoMsg;
    vector<fmlib::Command *> vecCmds; /* To store reference to commands inside the protobuf message */
    bool isComplete;                 /* Flag to determine if the request handling is complete */
    int st;                          /* Status code */
 
    if (!pConnection) {
        FM_LOG_ERROR("socket connection used by fabric manager API interface is not valid");
        FM_SYSLOG_ERR("socket connection used by fabric manager API interface is not valid");
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
        FM_LOG_ERROR("socket connection used by fabric manager API interface is not in active state");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return 0;
    }
 
    struct sockaddr_in remoteAddr =  pConnection->GetRemoteSocketAddr();
    std::string strAddr(inet_ntoa(remoteAddr.sin_addr));
 
    pFmServerRequest = pConnection->GetRequest(requestId);
    if (NULL == pFmServerRequest) {
        FM_LOG_ERROR("failed to get message/request from socket connection used by fabric manager API interface");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }    
 
    if (pFmServerRequest->MessageCount() != 1) {
        FM_LOG_ERROR("expected only single request message on socket connection used by fabric manager API interface");
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }
 
    // Get the message received corresponding to the request id
    pMessageRecvd = pFmServerRequest->GetNextMessage();
 
    if (NULL == pMessageRecvd) {
        FM_LOG_DEBUG("failed to get message for request id %d.", (int)requestId);
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    fmlibProtoMsg = new fmlib::Msg;

    if (true != fmlibProtoMsg->ParseFromArray((char *)pMessageRecvd->GetContent(), pMessageRecvd->GetLength())) {
        FM_LOG_ERROR("failed to parse and decode message received on socket connection used by fabric manager API interface");
        delete fmlibProtoMsg;
        delete pMessageRecvd;
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }
 
    /* The message is decoded and can be deleted as it won't be referenced again */
    delete pMessageRecvd;
    pMessageRecvd = NULL;
 
    fmlib::Command  *pCmd = (fmlib::Command*)&fmlibProtoMsg->cmd();

    if (pCmd == NULL) {
        FM_LOG_ERROR("failed to parse fabric manager API command from the received request message\n");
        delete fmlibProtoMsg;
        pConnection->CompleteRequest(requestId);
        pConnection->DecrReference();
        return -1;
    }

    processMessage(pCmd);
 
    //send FM Message back to the client
    st = sendReplyToClient(fmlibProtoMsg, pConnection, requestId, FM_MSG_PROTO_RESPONSE);
    if( st != 0) {
        FM_LOG_ERROR("failed to send fabric manager API command response message back to the requested client.");
    }
 
    delete fmlibProtoMsg;
    pConnection->CompleteRequest(requestId);
    pConnection->DecrReference();

    return 0;
}

 
void
GlobalFMLibCmdServer::processMessage(fmlib::Command *pCmd)  
{
    fmlib::CmdType cmdtype = pCmd->cmdtype();
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    fmReturn_t ret;
    void *apiCommandInfo = (void*)pCmd->arg().blob().c_str();
 
    switch (cmdtype) {
        case fmlib::FM_GET_SUPPORTED_PARTITIONS:
            {
                fm_msg_get_fabric_partition_t *getFabricPartition = (fm_msg_get_fabric_partition_t*) apiCommandInfo;
                if (getFabricPartition->version != fm_msg_get_fabric_partition_version) {
                    FM_LOG_ERROR("FM Get Supported Partition version mismatch detected. passed version: %X, internal version: %X",
                                 pCmd->version(), fm_msg_get_fabric_partition_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->getSupportedFabricPartitions(getFabricPartition->fmFabricPartitionList);
                fmlib::CmdArg *arg = (fmlib::CmdArg*)&pCmd->arg();
                arg->set_blob(apiCommandInfo, sizeof(*getFabricPartition));
                break;
            }
        case fmlib::FM_ACTIVATE_PARTITION:
            {  
                fm_msg_activate_fabric_partition_t *activateFabricPartition = (fm_msg_activate_fabric_partition_t*) apiCommandInfo;
                if (activateFabricPartition->version != fm_msg_activate_fabric_partition_version) {
                    FM_LOG_ERROR("FM Activate Fabric Partition version mismatch detected. passed version: %X, internal version: %X",
                                 activateFabricPartition->version, fm_msg_activate_fabric_partition_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->activateFabricPartition(activateFabricPartition->partitionId);
                break;
            }
        case fmlib::FM_ACTIVATE_PARTITION_WITH_VFS:
            {  
                fm_msg_activate_fabric_partition_vfs_t *activateFabricPartition = (fm_msg_activate_fabric_partition_vfs_t*) apiCommandInfo;
                if (activateFabricPartition->version != fm_msg_activate_fabric_partition_vfs_version) {
                    FM_LOG_ERROR("FM Activate Fabric Partition With VFs version mismatch detected. passed version: %X, internal version: %X",
                                 activateFabricPartition->version, fm_msg_activate_fabric_partition_vfs_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->activateFabricPartitionWithVfs(activateFabricPartition->partitionId, activateFabricPartition->vfList, activateFabricPartition->numVfs);
                break;
            }
        case fmlib::FM_DEACTIVATE_PARTITION:
            {
                fm_msg_deactivate_fabric_partition_t *deactivateFabricPartition = (fm_msg_deactivate_fabric_partition_t*) apiCommandInfo;
                if (deactivateFabricPartition->version != fm_msg_deactivate_fabric_partition_version) {
                    FM_LOG_ERROR("FM Deactivate Fabric Partition version mismatch detected. passed version: %X, internal version: %X",
                                 deactivateFabricPartition->version, fm_msg_deactivate_fabric_partition_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->deactivateFabricPartition(deactivateFabricPartition->partitionId);
                break;
            }
        case fmlib::FM_SET_ACTIVATED_PARTITION_LIST:
            {
                fm_msg_set_activated_fabric_partition_list_t *setActivatedFabricPartitionList = (fm_msg_set_activated_fabric_partition_list_t*) apiCommandInfo;
                if (setActivatedFabricPartitionList->version != fm_msg_set_activated_fabric_partition_version) {
                    FM_LOG_ERROR("FM Set Activated Fabric Partition version mismatch detected. passed version: %X, internal version: %X",
                                 setActivatedFabricPartitionList->version, fm_msg_set_activated_fabric_partition_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->setActivatedFabricPartitions(setActivatedFabricPartitionList->fmActivatedFabricPartitionList);
                break;
            }
        case fmlib::FM_GET_LWLINK_FAILED_DEVICES:
            {
                fm_msg_get_lwlink_failed_devices_t *getLwlinkFailedDevs = (fm_msg_get_lwlink_failed_devices_t*) apiCommandInfo;
                if (getLwlinkFailedDevs->version != fm_msg_get_lwlink_failed_devices_version) {
                    FM_LOG_ERROR("FM Get Lwlink Failed Devices version mismatch detected. passed version: %X, internal version: %X",
                                 pCmd->version(), fm_msg_get_lwlink_failed_devices_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }

                fmResult = mpApiHandler->getLwlinkFailedDevices(getLwlinkFailedDevs->fmLwlinkFailedDevices);
                fmlib::CmdArg *arg = (fmlib::CmdArg*)&pCmd->arg();
                arg->set_blob(apiCommandInfo, sizeof(*getLwlinkFailedDevs));
                break;
            }
        case fmlib::FM_GET_UNSUPPORTED_PARTITIONS:
            {
                fm_msg_get_unsupported_fabric_partition_t *getUnsupportedPartition = (fm_msg_get_unsupported_fabric_partition_t*) apiCommandInfo;
                if (getUnsupportedPartition->version != fm_msg_get_unsupported_fabric_partition_version) {
                    FM_LOG_ERROR("FM Get Unsupported Partition version mismatch detected. passed version: %X, internal version: %X",
                                 pCmd->version(), fm_msg_get_unsupported_fabric_partition_version);
                    pCmd->set_status(FM_INT_ST_VERSION_MISMATCH);
                    return;
                }
                fmResult = mpApiHandler->getUnsupportedFabricPartitions(getUnsupportedPartition->fmUnsupportedFabricPartitionList);
                fmlib::CmdArg *arg = (fmlib::CmdArg*)&pCmd->arg();
                arg->set_blob(apiCommandInfo, sizeof(*getUnsupportedPartition));
                break;
            }
        default:
            break;        
    }
 
    pCmd->set_status(fmResult);
}
 
int 
GlobalFMLibCmdServer::sendReplyToClient(fmlib::Msg *fmlibProtoMsg, FmServerConnection *pConnection, 
                                        fm_request_id_t requestId, unsigned int msgType) 
{
    char *msgToSend;
    unsigned int msgLen;    
    FmSocketMessage fmReply;
    int st;
 
    msgLen = fmlibProtoMsg->ByteSize();
    msgToSend = new char [msgLen];
    fmlibProtoMsg->SerializeToArray(msgToSend, msgLen);
 
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

