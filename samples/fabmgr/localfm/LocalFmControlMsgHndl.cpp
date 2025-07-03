/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "fm_log.h"
#include "LocalFmControlMsgHndl.h"
#include "LocalFMGpuMgr.h"
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include <g_lwconfig.h>


LocalFMControlMsgHndl::LocalFMControlMsgHndl(LocalFabricManagerControl *pControl)
{
    int i, numOfQueues;
    workqueue_t *pWorkqueue;

    if (NULL == pControl) {
        std::ostringstream ss;
        ss << "control msg handler: creating object with invalid local fabric manager object";
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    mpControl = pControl;
    mGfmHeartbeatTimer = nullptr;

    // create a workqueue for each switch interface, for switch message processing
    numOfQueues = mpControl->getNumLwswitchInterface();

    for (i = 0; i < numOfQueues; i++) {
        LocalFMSwitchInterface *pSwitchIntf = mpControl->switchInterfaceAtIndex(i);
        if (pSwitchIntf == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object");
            continue;
        }
        uint32_t switchDevIndex = pSwitchIntf->getSwitchDevIndex();
        pWorkqueue = new workqueue_t;
        mvWorkqueue.insert(make_pair(switchDevIndex, pWorkqueue));
        if ( workqueue_init(pWorkqueue, 1) ) {
            std::ostringstream ss;
            ss << "failed to create worker threads for handling LWSwitch configuration requests";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
};

LocalFMControlMsgHndl::~LocalFMControlMsgHndl()
{
    // shutdown and free message processing workqueues
    int i, numOfQueues = mvWorkqueue.size();
    for ( i = 0; i < numOfQueues; i++ )
    {
        workqueue_t * pWorkqueue = mvWorkqueue.at( i );
        if (pWorkqueue )
        {
            workqueue_shutdown( pWorkqueue );
            delete pWorkqueue;
        }
    }
    if ( mGfmHeartbeatTimer != nullptr)
        delete mGfmHeartbeatTimer;
};

void
LocalFMControlMsgHndl::handleEvent( FabricManagerCommEventType eventType, uint32 nodeId )
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            FM_LOG_DEBUG("nodeId %d FM_EVENT_PEER_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            FM_LOG_DEBUG("nodeId %d FM_EVENT_PEER_FM_DISCONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            FM_LOG_DEBUG("nodeId %d FM_EVENT_GLOBAL_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            FM_LOG_DEBUG("nodeId %d FM_EVENT_GLOBAL_FM_DISCONNECT", nodeId);
            break;
        }
    }
}

void
LocalFMControlMsgHndl::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    //FM_LOG_DEBUG("message type %d", pFmMessage->type());
    int queueNum = -1;

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_SWITCH_PORT_CONFIG_REQ:
    {
        const lwswitch::switchPortConfigRequest &portConfigRequest = pFmMessage->portconfigrequest();
        if ( portConfigRequest.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( portConfigRequest.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_INGRESS_REQUEST_TABLE_REQ:
    {
        const lwswitch::switchPortRequestTable &IngReqRequest = pFmMessage->requesttablerequest();
        if ( IngReqRequest.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( IngReqRequest.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_INGRESS_RESPONSE_TABLE_REQ:
    {
        const lwswitch::switchPortResponseTable &IngRspRequest = pFmMessage->responsetablerequest();
        if ( IngRspRequest.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( IngRspRequest.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_GANGED_LINK_TABLE_REQ:
    {
        const lwswitch::switchPortGangedLinkTable &gangedLinkConfigRequest = pFmMessage->gangedlinktablerequest();
        if ( gangedLinkConfigRequest.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( gangedLinkConfigRequest.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_GPU_CONFIG_REQ:
    {
        dumpMessage(pFmMessage);
        handleGpuConfigReqMsg(pFmMessage);
        return;
    }
    case lwswitch::FM_GPU_ATTACH_REQ:
    {
        dumpMessage(pFmMessage);
        handleGpuAttachReqMsg(pFmMessage);
        return;
    }
    case lwswitch::FM_GPU_DETACH_REQ:
    {
        dumpMessage(pFmMessage);
        handleGpuDetachReqMsg(pFmMessage);
        return;
    }
    case lwswitch::FM_CONFIG_INIT_DONE_REQ:
    {
        dumpMessage(pFmMessage);
        handleConfigInitDoneReqMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_CONFIG_DEINIT_REQ:
    {
        dumpMessage(pFmMessage);
        handleConfigDeInitReqMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_HEARTBEAT:
    {
        handleHeartbeatMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_NODE_GLOBAL_CONFIG_REQ:
    {
        dumpMessage(pFmMessage);
        handleNodeGlobalConfigReqMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_SWITCH_DISABLE_LINK_REQ:
    {
        dumpMessage(pFmMessage);
        handleSwitchDisableLinkReqMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_REQ:
    {
        dumpMessage(pFmMessage);
        handleGpuSetDisabledLinkMaskReqMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_GPU_GET_GFID_REQ:
    {
        dumpMessage(pFmMessage);
        handleGetGfidReqMsg(pFmMessage);
        return;
    }
    case lwswitch::FM_GPU_CFG_GFID_REQ:
    {
        dumpMessage(pFmMessage);
        handleCfgGfidReqMsg(pFmMessage);
        return;
    }
    case lwswitch::FM_RMAP_TABLE_REQ:
    {
        const lwswitch::portRmapTableRequest &rmapReq = pFmMessage->rmaptablereq();
        if ( rmapReq.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( rmapReq.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_RID_TABLE_REQ:
    {
        const lwswitch::portRidTableRequest &ridReq = pFmMessage->ridtablereq();
        if ( ridReq.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( ridReq.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_RLAN_TABLE_REQ:
    {
        const lwswitch::portRlanTableRequest &rlanReq = pFmMessage->rlantablereq();
        if ( rlanReq.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( rlanReq.switchphysicalid() );
        }
        break;
    }
    case lwswitch::FM_DEGRADED_GPU_INFO:
    {
        dumpMessage(pFmMessage);
        handleDegradedGpuInfoMsg( pFmMessage );
        return;
    }
    case lwswitch::FM_DEGRADED_LWSWITCH_INFO:
    {
        dumpMessage(pFmMessage);
        handleDegradedSwitchInfoMsg( pFmMessage );
        return;
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case lwswitch::FM_MCID_TABLE_SET_REQ:
    {
        const lwswitch::mcidTableSetRequest &mcidReq = pFmMessage->mcidtablesetreq();
        if ( mcidReq.has_switchphysicalid() )
        {
            queueNum = mpControl->getSwitchDevIndex( mcidReq.switchphysicalid() );
        }
        break;
    }
#endif
    default:
        FM_LOG_ERROR("control msg handler: detected unknown message type %d", pFmMessage->type());
        return;
    }

    // for messages which are not handled directly, process them using the
    // associated worker thread
    if (queueNum < 0)
    {
        // we shouldn't land here with an invalid worker queue number
        FM_LOG_ERROR("control msg handler: invalid worker queue number to process switch config messages");
        return;
    }

    workqueue_t *pQueue = NULL;

    // do all the error/validation before allocating messages/job object

    // enqueue the message to be processed by work queue thread
    if ( mvWorkqueue.find(queueNum) == mvWorkqueue.end()) {
        FM_LOG_ERROR("control msg handler: worker queue number to process switch config messages %d is not found",
                     queueNum);
        return;
    }

    pQueue = mvWorkqueue[queueNum];
    if ( pQueue == NULL )
    {
        FM_LOG_ERROR("control msg handler: worker queue object for queue number %d is invalid to process switch config messages",
                      queueNum);
        return;
    }

    job_t *pJob = new job_t;
    // make a copy of the message, because pFmMessage will be
    // freed after return from this method
    // LocalFMControlMsgHndl::processMessage will free the copy
    lwswitch::fmMessage *pMsg = pFmMessage->New();
    pMsg->CopyFrom(*pFmMessage);

    FmMessageUserData_t *pUsrData = new FmMessageUserData_t;
    pJob->user_data = pUsrData;
    pUsrData->pFmMessage = pMsg;
    pUsrData->pHndl = this;

    pJob->job_function = LocalFMControlMsgHndl::processMessageJob;

    workqueue_add_job(pQueue, pJob);
}

void
LocalFMControlMsgHndl::processMessage( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG("message type %d", pFmMessage->type());

    FM_LOG_DEBUG("pthread_t %d", (int)pthread_self());

    dumpMessage(pFmMessage);

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_SWITCH_PORT_CONFIG_REQ:
        handleSWPortConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_INGRESS_REQUEST_TABLE_REQ:
        handleIngReqTblConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_INGRESS_RESPONSE_TABLE_REQ:
        handleIngRespTblConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_GANGED_LINK_TABLE_REQ:
        handleGangedLinkTblConfigReqMsg( pFmMessage );
        break;
    case lwswitch::FM_RMAP_TABLE_REQ:
        handlePortRmapConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_RID_TABLE_REQ:
        handlePortRidConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_RLAN_TABLE_REQ:
        handlePortRlanConfigReqMsg( pFmMessage );
        break;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case lwswitch::FM_MCID_TABLE_SET_REQ:
        handleMulticastIdRequestMsg( pFmMessage );
        break;
#endif
    default:
        FM_LOG_ERROR("control msg handler: switch config worker detected unknown config message type %d",
                     pFmMessage->type());
        break;
    }

    delete pFmMessage;
}

/**
 *  thread function to process FM control messages
 */
void
LocalFMControlMsgHndl::processMessageJob( job_t *pJob )
{
    FmMessageUserData_t *pUsrData = (FmMessageUserData_t *) pJob->user_data;
    pUsrData->pHndl->processMessage(pUsrData->pFmMessage);

    delete pUsrData;
    pUsrData = NULL;
    delete pJob;
}

void
LocalFMControlMsgHndl::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    FM_LOG_DEBUG("%s", msgText.c_str());
#endif
}

/**
 *  on LFM, handle FM_NODE_GLOBAL_CONFIG_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleNodeGlobalConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pFmResponse;
    lwswitch::nodeGlobalConfigResponse *pRespMsg;

    if ( !pFmMessage->has_globalconfigrequest() ) {
        FM_LOG_WARNING("control msg handler: received global config request message without required fields\n");
        return;
    }

    // Note: This message can be extended such that GFM can indicate the topology specific Switches
    // and GPUs expected on this node and LFM can validate it based on the actual detected devices.

    const lwswitch::nodeGlobalConfigRequest &nodeConfigMsg = pFmMessage->globalconfigrequest();
    mpControl->setLocalNodeId( nodeConfigMsg.localnodeid() );

    // prepare the response message for gfm
    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );
    pFmResponse->set_type( lwswitch::FM_NODE_GLOBAL_CONFIG_RSP );

    // Note: This response message can be extended to indicate LFM state and previous initialization
    // status etc.

    pRespMsg = new lwswitch::nodeGlobalConfigResponse;
    pRespMsg->set_status( lwswitch::CONFIG_SUCCESS );

    pFmResponse->set_allocated_globalconfigresponse( pRespMsg );

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_DEBUG("Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_SWITCH_PORT_CONFIG_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleSWPortConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int ret = 0, i, requestId;
    LocalFMSwitchInterface *interface;
    lwswitch::switchPortConfigResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_SWITCH_PORT_CONFIG ioctlParams;

    memset(&ioctlStruct, 0, sizeof(ioctlStruct));
    memset(&ioctlParams, 0, sizeof(ioctlParams));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchPortConfigResponse;
    pFmResponse->set_allocated_portconfigresponse( pResponse );
    pFmResponse->set_type( lwswitch::FM_SWITCH_PORT_CONFIG_RSP );

    const lwswitch::switchPortConfigRequest &configRequest = pFmMessage->portconfigrequest();

    for ( i = 0; i < configRequest.info_size(); i++ )
    {
        const lwswitch::switchPortInfo * info = &configRequest.info(i);
        const ::switchPortConfig *config = &info->config();

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( configRequest.switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        ioctlParams.portNum = info->port();
        switch ( config->type() )
        {
        case ::ACCESS_PORT_GPU:
            ioctlParams.type = CONNECT_ACCESS_GPU;
            break;
        case ::TRUNK_PORT_SWITCH:
            ioctlParams.type = CONNECT_TRUNK_SWITCH;
            break;
        case ::ACCESS_PORT_SWITCH:
            ioctlParams.type = CONNECT_ACCESS_SWITCH;
        break;
        default:
            FM_LOG_ERROR("control msg handler: invalid switch port type %d is specified for port config request",
                         pFmMessage->type());
            break;
        };

        ioctlParams.acCoupled = false;
        if ( config->has_phymode() && ( config->phymode() == AC_COUPLED ) )
        {
            ioctlParams.acCoupled = true;
        }
        
        ioctlParams.enableVC1 = false;      //default when only VC set 0 can be present
        if ( config->has_enablevcset1() ) 
        {
            if (config->enablevcset1() != 0) ioctlParams.enableVC1 = true;
        }

        ioctlParams.requesterLinkID = config->has_requesterlinkid() ? config->requesterlinkid() : 0;
        ioctlParams.requesterLanID = config->has_rlanid() ? config->rlanid() : 0;
        ioctlParams.count = config->has_maxtargetid() ? config->maxtargetid() : 0;

        ioctlStruct.type = IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        interface = mpControl->switchInterfaceAt( configRequest.switchphysicalid() );
        if ( !interface )
        {
            FM_LOG_ERROR("control msg handler: failed to get LWSwitch driver interface object for physical Id  %d.",
                        configRequest.switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        rc = interface->doIoctl( &ioctlStruct );
        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch port config response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_INGRESS_REQUEST_TABLE_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleIngReqTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int ret = 0, i, j, requestId, entryCount;
    LocalFMSwitchInterface *interface;
    lwswitch::switchPortRequestTableResponse * pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_INGRESS_REQUEST_TABLE ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchPortRequestTableResponse;
    pFmResponse->set_allocated_requesttableresponse( pResponse );
    pFmResponse->set_type( lwswitch::FM_INGRESS_REQUEST_TABLE_RSP );

    const lwswitch::switchPortRequestTable &configRequest = pFmMessage->requesttablerequest();

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        const lwswitch::switchPortRequestTableInfo *info = &configRequest.info(j);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            FM_LOG_ERROR("control msg handler: failed to get LWSwitch driver interface object for physical Id  %d.",
                          info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = info->port();

        for ( i = 0; i < info->entry_size(); i++ )
        {
            const ::ingressRequestTable entry = info->entry(i);

            if ( entryCount == 0 )
            {
                // reset the firstIndex when starting a new ioctl of
                // LWSWITCH_INGRESS_REQUEST_SIZE entries
                //
                // all entries in configRequest.info have contiguous index
                // set by the global fabric manager
                ioctlParams.firstIndex = entry.index();
            }

            if ( entry.has_vcmodevalid7_0() )
            {
                ioctlParams.entries[entryCount].vcModeValid7_0 = entry.vcmodevalid7_0();
            }
            if ( entry.has_vcmodevalid15_8() )
            {
                ioctlParams.entries[entryCount].vcModeValid15_8 = entry.vcmodevalid15_8();
            }
            if ( entry.has_vcmodevalid17_16() )
            {
                ioctlParams.entries[entryCount].vcModeValid17_16 = entry.vcmodevalid17_16();
            }
            if ( entry.has_routepolicy() )
            {
                ioctlParams.entries[entryCount].routePolicy = entry.routepolicy();
            }
            if ( entry.has_address() )
            {
                ioctlParams.entries[i].mappedAddress = ( entry.address() >> 34 );
            }
            if ( entry.has_entryvalid() )
            {
                ioctlParams.entries[entryCount].entryValid = entry.entryvalid();
            }

            entryCount++;
            if ( entryCount == LWSWITCH_INGRESS_REQUEST_ENTRIES_MAX )
            {
                ioctlParams.numEntries = entryCount;
                entryCount = 0;

                // issue the ioctl when there are max number of entries
                rc = interface->doIoctl( &ioctlStruct );
                if ( rc != FM_INT_ST_OK )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_INT_ST_OK ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch routing table config response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_INGRESS_RESPONSE_TABLE_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleIngRespTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int ret = 0, i, j, requestId, entryCount;
    LocalFMSwitchInterface *interface;
    lwswitch::switchPortResponseTableResponse * pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE ioctlParams;

    memset(&ioctlStruct, 0, sizeof(ioctlStruct));
    memset(&ioctlParams, 0, sizeof(ioctlParams));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchPortResponseTableResponse;
    pFmResponse->set_allocated_responsetableresponse( pResponse );
    pFmResponse->set_type( lwswitch::FM_INGRESS_RESPONSE_TABLE_RSP );

    const lwswitch::switchPortResponseTable &configRequest = pFmMessage->responsetablerequest();

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        const lwswitch::switchPortResponseTableInfo *info = &configRequest.info(j);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            FM_LOG_ERROR("control msg handler: failed to get LWSwitch driver interface object for physical Id  %d.",
                          info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = info->port();

        for ( i = 0; i < info->entry_size(); i++ )
        {
            const ::ingressResponseTable entry = info->entry(i);

            if ( entryCount == 0 )
            {
                // reset the firstIndex when starting a new ioctl of
                // LWSWITCH_INGRESS_RESPONSE_SIZE entries
                //
                // all entries in configRequest.info have contiguous index
                // set by the global fabric manager
                ioctlParams.firstIndex = entry.index();
            }

            if ( entry.has_vcmodevalid7_0() )
            {
                ioctlParams.entries[entryCount].vcModeValid7_0 = entry.vcmodevalid7_0();
            }
            if ( entry.has_vcmodevalid15_8() )
            {
                ioctlParams.entries[entryCount].vcModeValid15_8 = entry.vcmodevalid15_8();
            }
            if ( entry.has_vcmodevalid17_16() )
            {
                ioctlParams.entries[entryCount].vcModeValid17_16 = entry.vcmodevalid17_16();
            }
            if ( entry.has_routepolicy() )
            {
                ioctlParams.entries[entryCount].routePolicy = entry.routepolicy();
            }
            if ( entry.has_entryvalid() )
            {
                ioctlParams.entries[entryCount].entryValid = entry.entryvalid();
            }

            entryCount++;
            if ( entryCount == LWSWITCH_INGRESS_RESPONSE_ENTRIES_MAX )
            {
                ioctlParams.numEntries = entryCount;
                entryCount = 0;

                // issue the ioctl when there are max number of entries
                rc = interface->doIoctl( &ioctlStruct );
                if ( rc != FM_INT_ST_OK ) {
                    break;
                }
            }
        }

        if ( ( rc == FM_INT_ST_OK ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch routing table config response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GANGED_LINK_TABLE_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleGangedLinkTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int ret = 0, i, j, index, requestId;
    LocalFMSwitchInterface *interface;
    lwswitch::switchPortGangedLinkTableResponse * pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_GANGED_LINK_TABLE ioctlParams;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchPortGangedLinkTableResponse;
    pFmResponse->set_allocated_gangedlinktableresponse( pResponse );
    pFmResponse->set_type( lwswitch::FM_GANGED_LINK_TABLE_RSP );

    const lwswitch::switchPortGangedLinkTable &configRequest = pFmMessage->gangedlinktablerequest();
    const lwswitch::switchPortGangedLinkTableInfo *info = NULL;
    lwswitch::configResponse *instanceResponse = NULL;

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        info = &configRequest.info(j);

        instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            FM_LOG_ERROR("control msg handler: failed to get LWSwitch driver interface object for physical Id  %d.",
                          info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // construct the ioctl on the new port
        ioctlStruct.type = IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        memset( &ioctlParams, 0, sizeof(LWSWITCH_SET_GANGED_LINK_TABLE) );
        ioctlParams.link_mask = 1 << info->port();

        if  ( info->has_table() )
        {
            for ( index = 0; index < info->table().data_size(); index++ )
            {
                ioctlParams.entries[ index ] = info->table().data(index);
            }
        }

        // send the ioctl from the previous port
        rc = interface->doIoctl( &ioctlStruct );
        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch routing table config response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_CONFIG_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleGpuConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int i;
    lwswitch::gpuConfigResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuConfigResponse;
    pFmResponse->set_allocated_gpuconfigrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_GPU_CONFIG_RSP );

    const lwswitch::gpuConfigRequest &configRequest = pFmMessage->gpuconfigreq();

    for ( i = 0; i < configRequest.info_size(); i++ )
    {
        const lwswitch::gpuInfo * pInfo = &configRequest.info(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( pInfo->gpuphysicalid() );

        if ( configRequest.has_partitionid() )
            instanceResponse->set_partitionid( configRequest.partitionid() );
        else
            instanceResponse->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

        if ( !pInfo->has_uuid() )
        {
             FM_LOG_ERROR("control msg handler: GPU uuid information is missing in GPU config request message");
             rc = FM_INT_ST_ILWALID_GPU;
        }
        else
        {
            FMUuid_t fmUuid;
            strncpy(fmUuid.bytes, pInfo->uuid().c_str(), FM_UUID_BUFFER_SIZE - 1);
            FMPciInfo_t fmGpuPciInfo;
            mpControl->getGpuPciInfo(fmUuid, fmGpuPciInfo);

            // Skip programming of GPA address in VGPU mode
            if ( (mpControl->getFabricMode() != FM_MODE_VGPU) && pInfo->has_fabricaddressbase() )
            {
                rc = mpControl->mFMGpuMgr->setGpuFabricGPA(fmUuid, pInfo->fabricaddressbase());
                if( FM_INT_ST_OK != rc )
                {
                    FM_LOG_ERROR("GPU physical address assignment failed for GPU uuid %s pci bus id %s with error %d",
                                 fmUuid.bytes, fmGpuPciInfo.busId, rc);
                }
            }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

            // Skip programming of GPA address in VGPU mode
            if ( (mpControl->getFabricMode() != FM_MODE_VGPU) && pInfo->has_gpaegmaddressbase() && pInfo->has_gpaegmaddressrange() )
            {
                rc = mpControl->mFMGpuMgr->setGpuFabricGPAEgm(fmUuid, pInfo->gpaaddressbase());
                if( FM_INT_ST_OK != rc )
                {
                    FM_LOG_ERROR("GPU physical EGM address range assignment failed for GPU uuid %s pci bus id %s with error %d",
                                 fmUuid.bytes, fmGpuPciInfo.busId, rc);
                }
            }
#endif

            // Skip programming of GPA address in VGPU mode
            if ( (mpControl->getFabricMode() != FM_MODE_VGPU) && pInfo->has_gpaaddressbase() && pInfo->has_gpaaddressrange() )
            {
                rc = mpControl->mFMGpuMgr->setGpuFabricGPA(fmUuid, pInfo->gpaaddressbase());
                if( FM_INT_ST_OK != rc )
                {
                    FM_LOG_ERROR("GPU physical address range assignment failed for GPU uuid %s pci bus id %s with error %d",
                                 fmUuid.bytes, fmGpuPciInfo.busId, rc);
                }
            }

            if ( pInfo->has_flaaddressbase() && pInfo->has_flaaddressrange() )
            {
                rc = configRequest.config() ?
                    mpControl->mFMGpuMgr->setGpuFabricFLA(fmUuid, pInfo->flaaddressbase(), pInfo->flaaddressrange()) :
                    mpControl->mFMGpuMgr->clearGpuFabricFLA(fmUuid, pInfo->flaaddressbase(), pInfo->flaaddressrange());

                if( FM_INT_ST_OK != rc )
                {
                    FM_LOG_ERROR("GPU fabric linear address assignment failed for GPU uuid %s pci bus id %s with error %d",
                                 fmUuid.bytes, fmGpuPciInfo.busId, rc);
                }
            }
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send GPU config response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_ATTACH_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleGpuAttachReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int i;
    lwswitch::gpuAttachResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuAttachResponse;
    pFmResponse->set_allocated_gpuattachrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_GPU_ATTACH_RSP );

    const lwswitch::gpuAttachRequest &configRequest = pFmMessage->gpuattachreq();

    //
    // before trying to attach the GPUs, forcefully update the RM API layer to
    // re-fetch the newly probed/attached GPUs
    //
    // Note: This is a temporary work around as RM and RM API layer don't have
    // mechanism to notify GPU hot-attach
    //
    mpControl->refreshRmLibProbedGpuInfo();

    for ( i = 0; i < configRequest.info_size(); i++ )
    {
        const lwswitch::gpuAttachRequestInfo *pInfo = &configRequest.info(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();

        if ( pInfo->has_gpuphysicalid() )
        {
            instanceResponse->set_devicephysicalid( pInfo->gpuphysicalid() );
        }

        if ( pInfo->has_uuid() )
        {
            instanceResponse->set_uuid( pInfo->uuid().c_str() );
        }

        if ( configRequest.has_partitionid() )
            instanceResponse->set_partitionid( configRequest.partitionid() );
        else
            instanceResponse->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

        char *uuid = (char*)pInfo->uuid().c_str();
        FMUuid_t gpuUuid;
        strncpy(gpuUuid.bytes, uuid, sizeof(gpuUuid.bytes)-1); 

        bool registerEvent = false;
        if ( pInfo->has_registerevent() && ( pInfo->registerevent() == true ) )
        {
            registerEvent = true;
        }

        fmResult = mpControl->attachGpu(gpuUuid, registerEvent);
        if ( fmResult == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send GPU attach response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_DETACH_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleGpuDetachReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int i;
    lwswitch::gpuDetachResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuDetachResponse;
    pFmResponse->set_allocated_gpudetachrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_GPU_DETACH_RSP );

    const lwswitch::gpuDetachRequest &configRequest = pFmMessage->gpudetachreq();

    // check to see if we need to detach all GPUs or only GPUs in a certain partition
    int size = configRequest.info_size() == 0 ? 1 : configRequest.info_size();

    for ( i = 0; i < size; i++ )
    {
        lwswitch::configResponse *instanceResponse = pResponse->add_response();

        if ( configRequest.has_partitionid() )
            instanceResponse->set_partitionid( configRequest.partitionid() );
        else
            instanceResponse->set_partitionid( ILWALID_FABRIC_PARTITION_ID );


        /*
            Here we detach All Gpus instead of detaching the GPUs one by one.
            Cases where these are needed:
            i)  If there is some attach issues, then GlobalFMErrorHndl can just detach all the GPUs
            ii) Detaching all GPUs at the end of FM Initialization in Shared fabric Mode
        */
        if (configRequest.info_size() == 0) {
            fmResult = mpControl->detachAllGpus();
        }

        // Per GPU detach
        else {
            FMUuid_t gpuUuid;
            const lwswitch::gpuDetachRequestInfo * pInfo = &configRequest.info(i);
            
            char *uuid = (char*)pInfo->uuid().c_str();
            
            if ( pInfo->has_gpuphysicalid() )
            {
                instanceResponse->set_devicephysicalid( pInfo->gpuphysicalid() );
            }

            if ( pInfo->has_uuid() )
            {
                instanceResponse->set_uuid( pInfo->uuid().c_str() );
            }
            strncpy(gpuUuid.bytes, uuid, sizeof(gpuUuid.bytes)-1); 

            bool unregisterEvent = false;
            if ( pInfo->has_unregisterevent() && ( pInfo->unregisterevent() == true ) )
            {
                unregisterEvent = true;
            }

            fmResult = mpControl->detachGpu(gpuUuid, unregisterEvent);
        }

        if ( fmResult == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send GPU detach response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_CONFIG_INIT_DONE_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleConfigInitDoneReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    lwswitch::configInitDoneRsp *pInitDoneRsp;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    bool initReturn;

    // notify localFM control to take appropriate actions    
    initReturn = mpControl->onConfigInitDoneReqRcvd();

    // prepare the response message    
    pInitDoneRsp = new lwswitch::configInitDoneRsp();

    if ( initReturn ) {
        pInitDoneRsp->set_status( lwswitch::CONFIG_SUCCESS );
    } else {
        pInitDoneRsp->set_status( lwswitch::CONFIG_FAILURE );
    }

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );
    pFmResponse->set_allocated_initdonersp( pInitDoneRsp );
    pFmResponse->set_type( lwswitch::FM_CONFIG_INIT_DONE_RSP );

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("control msg handler: failed to send config initialization done response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_CONFIG_DEINIT_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleConfigDeInitReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    lwswitch::configDeInitRsp *pDeInitRsp;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    bool deInitReturn;

    // notify localFM control to take appropriate actions
    deInitReturn = mpControl->onConfigDeInitReqRcvd();

    // prepare the response message    
    pDeInitRsp = new lwswitch::configDeInitRsp();
    if ( deInitReturn ) {
        pDeInitRsp->set_status( lwswitch::CONFIG_SUCCESS );
    } else {
        pDeInitRsp->set_status( lwswitch::CONFIG_FAILURE );
    }

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );
    pFmResponse->set_allocated_deinitrsp( pDeInitRsp );
    pFmResponse->set_type( lwswitch::FM_CONFIG_DEINIT_RSP );

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("control msg handler: failed to send config deinitialization done response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_HEARTBEAT message
 */
void
LocalFMControlMsgHndl::handleHeartbeatMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int i;
    const lwswitch::heartbeat *pRequest = &(pFmMessage->heartbeat());
    lwswitch::heartbeatAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //heartbeat ack to send back to GFM

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_type( lwswitch::FM_HEARTBEAT_ACK );
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::heartbeatAck;
    pResponse->set_nodeid( pRequest->nodeid() );
    pFmResponse->set_allocated_heartbeatack( pResponse );

    if ( mpControl )
    {
        // send the heartbeat ack to gfm
        rc = mpControl->SendMessageToGfm( pFmResponse, false );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("control msg handler: failed to send hearbeat message");
        }
    }
    // Once the first heartbeat is received the LFM also starts a timer. If this timer expires
    // it implies GFM heartbeat are no longer seen for HEARTBEAT_THRESHOLD intervals and LFM
    // will take appropriate actions for disabling FM session and stopping IMEX processing
    // This is needed as once the GFM socket connection is disconnected, LFM won't get messages from GFM
    if ( mGfmHeartbeatTimer == nullptr )
    {
        FM_LOG_DEBUG( "Received first heartbeat from GFM" );
        mGfmHeartbeatTimer = new FMTimer( LocalFMControlMsgHndl::gfmHeartbeatTimeoutHandler, this );
        mGfmHeartbeatTimer->start( HEARTBEAT_INTERVAL * HEARTBEAT_THRESHOLD );
    }
    else
    {
        mGfmHeartbeatTimer->restart();
    }
    delete pFmResponse;
}

/**
 *  on LFM, handle FM_SWITCH_DISABLE_LINK_REQ message
 */
void
LocalFMControlMsgHndl::handleSwitchDisableLinkReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    LocalFMSwitchInterface *interface = NULL;
    const lwswitch::switchDisableLinkRequest disableReqMsg = pFmMessage->switchdisablelinkreq();
    lwswitch::configStatus retStatus = lwswitch::CONFIG_SUCCESS;

    interface = mpControl->switchInterfaceAt( disableReqMsg.switchphysicalid() );
    if ( interface )
    {
        for ( int idx = 0; idx < disableReqMsg.portnum_size(); idx++ )
        {
            uint32 portNum = disableReqMsg.portnum( idx );
            switchIoctl_t ioctlStruct;
            LWSWITCH_UNREGISTER_LINK_PARAMS ioctlParams;
            memset( &ioctlStruct, 0, sizeof(ioctlStruct) );
            memset( &ioctlParams, 0, sizeof(ioctlParams) );
            ioctlStruct.type = IOCTL_LWSWITCH_UNREGISTER_LINK;
            ioctlParams.portNum = portNum;
            ioctlStruct.ioctlParams = &ioctlParams;
            ioctlStruct.paramSize = sizeof(ioctlParams);
            rc = interface->doIoctl( &ioctlStruct );
            if ( rc != FM_INT_ST_OK )
            {
                FMPciInfo_t pciInfo = interface->getSwtichPciInfo();
                FM_LOG_ERROR("disabling LWLink switch port %d failed for switch physical id %d pci bus id %s",
                              portNum, disableReqMsg.switchphysicalid(), pciInfo.busId);
                retStatus = lwswitch::CONFIG_FAILURE;
                break;
            }
        }
        // update enabled port mask information after links are disabled
        interface->updateEnabledPortMask();
    } else
    {
        FM_LOG_ERROR("control msg handler: failed to get LWSwitch driver interface object for physical Id  %d.",
                      disableReqMsg.switchphysicalid());
        retStatus = lwswitch::CONFIG_FAILURE;
    }

    // send the response back to gfm
    lwswitch::switchDisableLinkResponse *pDisableRspMsg = new lwswitch::switchDisableLinkResponse();
    lwswitch::fmMessage *pFmResponse = new lwswitch::fmMessage();

    pDisableRspMsg->set_status( retStatus );
    pFmResponse->set_requestid( pFmMessage->requestid() );
    pFmResponse->set_allocated_switchdisablelinkrsp( pDisableRspMsg );
    pFmResponse->set_type( lwswitch::FM_SWITCH_DISABLE_LINK_RSP );

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch port disable response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::handleGpuSetDisabledLinkMaskReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    lwswitch::configStatus retStatus = lwswitch::CONFIG_SUCCESS;
    lwswitch::gpuSetDisabledLinkMaskResponse *pMaskRespMsg = new lwswitch::gpuSetDisabledLinkMaskResponse();
    //
    // GPU LWLink disabled mask should be written before the GPU is initialized/attached by any RM client.
    //

    //
    // Note: This is very specific to Shared Fabric Mode, which assumes that the GPUs are not attached by
    // any RM client once they are hot plugged into the ServiceVM as part of partition activation.
    //

    // set disabled link mask for each GPU
    const lwswitch::gpuSetDisabledLinkMaskRequest &disableMaskReqMsg = pFmMessage->gpusetdisabledlinkmaskreq();

    for ( int idx = 0; idx < disableMaskReqMsg.gpuinfo_size(); idx++ )
    {
        const lwswitch::gpuDisabledLinkMaskInfoMsg &gpuInfoMsg = disableMaskReqMsg.gpuinfo( idx );
        // we must have gpu uuid to set the disabled link mask
        if ( !gpuInfoMsg.has_uuid() )
        {
            FM_LOG_ERROR("control msg handler: GPU uuid information is missing in GPU link disable request message");
            retStatus = lwswitch::CONFIG_FAILURE;
            break;
        }
        char *uuid = (char*)gpuInfoMsg.uuid().c_str();
        FMUuid_t gpuUuid;
        memset(gpuUuid.bytes, 0, sizeof(gpuUuid.bytes));
        strncpy(gpuUuid.bytes, uuid, sizeof(gpuUuid.bytes)-1); 
        uint32 disabledMask = gpuInfoMsg.disablemask();
        rc = mpControl->mFMGpuMgr->setGpuLWLinkInitDisabledMask(gpuUuid, disabledMask);
        if (rc != FM_INT_ST_OK) {
            // error already logged
            pMaskRespMsg->set_uuid( uuid );
            retStatus = lwswitch::CONFIG_FAILURE;
            break;
        }

    }

    // generate the final response to GlobalFM
    lwswitch::fmMessage *pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );
    pFmResponse->set_type( lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_RSP );
    pMaskRespMsg->set_status( retStatus );
    pFmResponse->set_allocated_gpusetdisabledlinkmaskrsp( pMaskRespMsg );

    if ( disableMaskReqMsg.has_partitionid() )
    {
        pMaskRespMsg->set_partitionid(disableMaskReqMsg.partitionid());
    }
    else
    {
        pMaskRespMsg->set_partitionid( ILWALID_FABRIC_PARTITION_ID );
    }

    // send the response back to GlobalFM
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send GPU LWLink initialization disabled config response message");
    }
    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_GET_GFID_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleGetGfidReqMsg(lwswitch::fmMessage *pFmMessage)
{
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    lwswitch::gpuGetGfidResponse *pResponse;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    FMIntReturn_t rc;
    int i;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid(pFmMessage->requestid());

    pResponse = new lwswitch::gpuGetGfidResponse;
    pFmResponse->set_allocated_gpugetgfidrsp(pResponse);
    pFmResponse->set_type(lwswitch::FM_GPU_GET_GFID_RSP);

    const lwswitch::gpuGetGfidRequest &pRequest = pFmMessage->gpugetgfidreq();
    pResponse->set_status(lwswitch::CONFIG_SUCCESS);

    if (pRequest.has_partitionid())
        pResponse->set_partitionid(pRequest.partitionid());

    for (i = 0; i < pRequest.info_size(); i++)
    {
        const lwswitch::gpuGetGfidRequestInfo *pInfo = &pRequest.info(i);
        const lwswitch::devicePciInfo vf = pInfo->vf();
        uint32_t gfid, gfidMask;
        FMPciInfo_t vfInfo;

        lwswitch::gpuGetGfidResponseInfo *instanceResponse = pResponse->add_info();

        if (pInfo->has_physicalid())
        {
            instanceResponse->set_physicalid(pInfo->physicalid());
        }

        if (pInfo->has_uuid())
        {
            instanceResponse->set_uuid(pInfo->uuid().c_str());
        }

        vfInfo.domain = (unsigned int ) vf.domain();
        vfInfo.bus = (unsigned int ) vf.bus();
        vfInfo.device = (unsigned int ) vf.device();
        vfInfo.function = (unsigned int ) vf.function();

        char *uuid = (char*)pInfo->uuid().c_str();
        FMUuid_t gpuUuid;
        strncpy(gpuUuid.bytes, uuid, sizeof(gpuUuid.bytes)-1); 

        fmResult = mpControl->getGpuGfid(gpuUuid, vfInfo, gfid, gfidMask);
        if (fmResult != FM_INT_ST_OK) {
            pResponse->set_status(lwswitch::CONFIG_FAILURE);
            break;
        }

        instanceResponse->set_gfid (gfid);
        instanceResponse->set_gfidmask (gfidMask);
    }

    // send the response back to gfm
    rc = SendMessageToGfm(pFmResponse, false);
    if (rc != FM_INT_ST_OK)
    {
        FM_LOG_ERROR("control msg handler: failed to send get gfid response message");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_CFG_GFID_REQ message from GFM
 */
void
LocalFMControlMsgHndl::handleCfgGfidReqMsg(lwswitch::fmMessage *pFmMessage)
{
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    lwswitch::gpuCfgGfidResponse *pResponse;
    FMIntReturn_t fmResult = FM_INT_ST_OK;
    FMIntReturn_t rc;
    int i;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid(pFmMessage->requestid());

    pResponse = new lwswitch::gpuCfgGfidResponse;
    pFmResponse->set_allocated_gpucfggfidrsp(pResponse);
    pFmResponse->set_type(lwswitch::FM_GPU_CFG_GFID_RSP);

    const lwswitch::gpuCfgGfidRequest &pRequest = pFmMessage->gpucfggfidreq();
    pResponse->set_status(lwswitch::CONFIG_SUCCESS);

    if (pRequest.has_partitionid())
        pResponse->set_partitionid(pRequest.partitionid());

    for (i = 0; i < pRequest.info_size(); i++)
    {
        const lwswitch::gpuCfgGfidRequestInfo *pInfo = &pRequest.info(i);

        char *uuid = (char*)pInfo->uuid().c_str();
        FMUuid_t gpuUuid;
        strncpy(gpuUuid.bytes, uuid, sizeof(gpuUuid.bytes)-1); 

        fmResult = mpControl->configGpuGfid(gpuUuid, pInfo->gfid(), pRequest.activate());
        if ( fmResult != FM_INT_ST_OK ) {
            pResponse->set_status(lwswitch::CONFIG_FAILURE);
            break;
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm(pFmResponse, false);
    if (rc != FM_INT_ST_OK)
    {
        FM_LOG_ERROR("control msg handler: failed to send cfg gfid response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::handlePortRmapConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int ret = 0, i, j, requestId, entryCount;
    LocalFMSwitchInterface *interface;
    lwswitch::portRmapTableResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_REMAP_POLICY ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::portRmapTableResponse;
    pFmResponse->set_allocated_rmaptablersp( pResponse );
    pFmResponse->set_type( lwswitch::FM_RMAP_TABLE_RSP );

    const lwswitch::portRmapTableRequest &configRequest = pFmMessage->rmaptablereq();

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        const lwswitch::portRmapTableInfo *info = &configRequest.info(j);

        RemapTable remapTable = info->has_table() ? info->table() : NORMAL_RANGE;

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        if (interface == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d", info->switchphysicalid());
            instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
            continue;
        }        

        ioctlStruct.type = IOCTL_LWSWITCH_SET_REMAP_POLICY;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = info->port();

        // set the corresponding driver remap table
        switch (remapTable) {
        case NORMAL_RANGE:
            ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
            break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case EXTENDED_RANGE_A:
            ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_EXTA;
            break;

        case EXTENDED_RANGE_B:
            ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_EXTB;
            break;

        case MULTICAST_RANGE:
            ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_MULTICAST;
            break;
#endif

        default:
            ioctlParams.tableSelect = LWSWITCH_TABLE_SELECT_REMAP_PRIMARY;
            break;
        }

        for ( i = 0; i < info->entry_size(); i++ )
        {
            const ::rmapPolicyEntry entry = info->entry(i);

            if ( entryCount == 0 )
            {
                // reset the firstIndex when starting a new ioctl of
                // LWSWITCH_REMAP_POLICY_ENTRIES_MAX entries
                //
                // all entries in configRequest.info have contiguous index
                // set by the global fabric manager
                ioctlParams.firstIndex = entry.index();
            }

            // set up Rmap Policy Entry
            memset(&ioctlParams.remapPolicy[entryCount], 0, sizeof(LWSWITCH_REMAP_POLICY_ENTRY));

            if ( entry.has_entryvalid() )
            {
                ioctlParams.remapPolicy[entryCount].entryValid = entry.entryvalid();
            }

            if ( entry.has_targetid() )
            {
                ioctlParams.remapPolicy[entryCount].targetId = entry.targetid();
            }

            if ( entry.has_irlselect() )
            {
                ioctlParams.remapPolicy[entryCount].irlSelect = entry.irlselect();
            }

            if ( entry.has_address() )
            {
                ioctlParams.remapPolicy[entryCount].address = entry.address();
            }

            if ( entry.has_reqcontextmask() )
            {
                ioctlParams.remapPolicy[entryCount].reqCtxMask = entry.reqcontextmask();
            }

            if ( entry.has_reqcontextchk() )
            {
                ioctlParams.remapPolicy[entryCount].reqCtxChk = entry.reqcontextchk();
            }

            if ( entry.has_reqcontextrep() )
            {
                ioctlParams.remapPolicy[entryCount].reqCtxRep = entry.reqcontextrep();
            }

            if ( entry.has_addressoffset() )
            {
                ioctlParams.remapPolicy[entryCount].addressOffset = entry.addressoffset();
            }

            if ( entry.has_addressbase() )
            {
                ioctlParams.remapPolicy[entryCount].addressBase = entry.addressbase();
            }

            if ( entry.has_addresslimit() )
            {
                ioctlParams.remapPolicy[entryCount].addressLimit = entry.addresslimit();
            }

            if (entry.has_remapflags() )
            {
                uint32_t driverFlags;
                colwertFmToDriverRemapPolicyFlags(entry.remapflags(), driverFlags);
                ioctlParams.remapPolicy[entryCount].flags = driverFlags;
            }

            /*
                   LR TODO
                   The following are not defined in ioctl params check with Paul
                   optional uint32  routingFunction     = 12; is this flags?
                   optional uint32  p2rSwizEnable       = 14;
                   optional uint32  mult2               = 15;
                   optional uint32  planeSelect         = 16;
             */

            entryCount++;
            if ( entryCount == LWSWITCH_REMAP_POLICY_ENTRIES_MAX )
            {
                ioctlParams.numEntries = entryCount;
                entryCount = 0;

                // issue the ioctl when there are max number of entries
                rc = interface->doIoctl( &ioctlStruct );
                if ( rc != FM_INT_ST_OK )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_INT_ST_OK ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch route map table config response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::handlePortRidConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int ret = 0, i, j, p, requestId, entryCount;
    LocalFMSwitchInterface *interface;
    lwswitch::portRidTableResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_ROUTING_ID ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::portRidTableResponse;
    pFmResponse->set_allocated_ridtablersp( pResponse );
    pFmResponse->set_type( lwswitch::FM_RID_TABLE_RSP );

    const lwswitch::portRidTableRequest &configRequest = pFmMessage->ridtablereq();

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        const lwswitch::portRidTableInfo *info = &configRequest.info(j);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        if (interface == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d", info->switchphysicalid());
            instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_ROUTING_ID;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = info->port();

        for ( i = 0; i < info->entry_size(); i++ )
        {
            const ::ridRouteEntry entry = info->entry(i);

            if ( entryCount == 0 )
            {
                // reset the firstIndex when starting a new ioctl of
                // LWSWITCH_ROUTING_ID_ENTRIES_MAX entries
                //
                // all entries in configRequest.info have contiguous index
                // set by the global fabric manager
                ioctlParams.firstIndex = entry.index();
            }

            // set up Routing ID Entry
            memset(&ioctlParams.routingId[entryCount], 0, sizeof(LWSWITCH_ROUTING_ID_ENTRY));

            if ( entry.has_valid() )
            {
                ioctlParams.routingId[entryCount].entryValid = entry.valid();
            }

            if ( entry.has_rmod() )
            {
                /*
                   From IAS
                   rmod[6]: Use the results of the RLAN Route RAM
                 */
                ioctlParams.routingId[entryCount].useRoutingLan = (entry.rmod() >> 6) & 1;

                /*
                   From IAS
                   rmod[9]: Enable error response generation on ping request packets with
                   irl[1:0] = 2'b11. 1: pass ping requests through NXbar
                 */
                ioctlParams.routingId[entryCount].enableIrlErrResponse = (entry.rmod() >> 9) & 1;
            }

            // set up portlist
            ioctlParams.routingId[entryCount].numEntries = entry.portlist_size();
            for ( p = 0; p < entry.portlist_size(); p++ )
            {
                const ::routePortList portList = entry.portlist(p);
                if ( portList.has_vcmap() )
                {
                    ioctlParams.routingId[entryCount].portList[p].vcMap = portList.vcmap();
                }

                if ( portList.has_portindex() )
                {
                    ioctlParams.routingId[entryCount].portList[p].destPortNum = portList.portindex();
                }
            }

            entryCount++;
            if ( entryCount == LWSWITCH_ROUTING_ID_ENTRIES_MAX )
            {
                ioctlParams.numEntries = entryCount;
                entryCount = 0;

                // issue the ioctl when there are max number of entries
                rc = interface->doIoctl( &ioctlStruct );
                if ( rc != FM_INT_ST_OK )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_INT_ST_OK ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch route id config response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::handlePortRlanConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    int ret = 0, i, j, p, requestId, entryCount;
    LocalFMSwitchInterface *interface;
    lwswitch::portRlanTableResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_ROUTING_LAN ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    memset(&ioctlStruct, 0, sizeof(ioctlStruct));

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::portRlanTableResponse;
    pFmResponse->set_allocated_rlantablersp( pResponse );
    pFmResponse->set_type( lwswitch::FM_RLAN_TABLE_RSP );

    const lwswitch::portRlanTableRequest &configRequest = pFmMessage->rlantablereq();

    for ( j = 0; j < configRequest.info_size(); j++ )
    {
        const lwswitch::portRlanTableInfo *info = &configRequest.info(j);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        if (interface == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d", info->switchphysicalid());
            instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_ROUTING_LAN;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);
        ioctlParams.portNum = info->port();

        for ( i = 0; i < info->entry_size(); i++ )
        {
            const ::rlanRouteEntry entry = info->entry(i);

            if ( entryCount == 0 )
            {
                // reset the firstIndex when starting a new ioctl of
                // LWSWITCH_ROUTING_LAN_ENTRIES_MAX entries
                //
                // all entries in configRequest.info have contiguous index
                // set by the global fabric manager
                ioctlParams.firstIndex = entry.index();
            }

            // set up Routing LAN Entry
            memset(&ioctlParams.routingLan[entryCount], 0, sizeof(LWSWITCH_ROUTING_LAN_ENTRY));

            if ( entry.has_valid() )
            {
                ioctlParams.routingLan[entryCount].entryValid = entry.valid();
            }

            // set up group list
            ioctlParams.routingLan[entryCount].numEntries = entry.grouplist_size();
            for ( p = 0; p < entry.grouplist_size(); p++ )
            {
                const ::rlanGroupSel groupList = entry.grouplist(p);
                if ( groupList.has_groupselect() )
                {
                    ioctlParams.routingLan[entryCount].portList[p].groupSelect = groupList.groupselect();
                }

                if ( groupList.has_groupsize() )
                {
                    ioctlParams.routingLan[entryCount].portList[p].groupSize = groupList.groupsize();
                }
            }

            entryCount++;
            if ( entryCount == LWSWITCH_ROUTING_LAN_ENTRIES_MAX )
            {
                ioctlParams.numEntries = entryCount;
                entryCount = 0;

                // issue the ioctl when there are max number of entries
                rc = interface->doIoctl( &ioctlStruct );
                if ( rc != FM_INT_ST_OK )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_INT_ST_OK ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_INT_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send switch routing table config response message");
    }
    delete pFmResponse;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

bool
LocalFMControlMsgHndl::mcIdEntryToIoctlParams( uint32_t portNum, bool noDynRsp,
                                               const lwswitch::MulticastTableEntry &entry,
                                               LWSWITCH_SET_MC_RID_TABLE_PARAMS &ioctlParams )
{
    memset(&ioctlParams, 0, sizeof(LWSWITCH_SET_MC_RID_TABLE_PARAMS));

    ioctlParams.portNum = portNum;
    ioctlParams.index = entry.index();
    ioctlParams.extendedTable = entry.extendedtbl();
    ioctlParams.entryValid = entry.entryvalid();

    if (entry.has_extendedtblindex()) {
        ioctlParams.extendedValid = true;
        ioctlParams.extendedPtr = entry.extendedtblindex();
    }

    ioctlParams.numSprayGroups = (uint32_t)entry.sprays_size();
    ioctlParams.mcSize = entry.sprays(0).ports_size();

    uint32_t portIndex = 0;

    for ( int i = 0; i < entry.sprays_size(); i++ ) {
        const lwswitch::MulticastPortList &portList = entry.sprays(i);

        if ( (uint32_t)portList.ports_size() != ioctlParams.mcSize ) {
            FM_LOG_ERROR("multicast group index %d port list %d has %d number of ports, which is different from %d.",
                         ioctlParams.index, i, portList.ports_size(), ioctlParams.mcSize);
            return false;
        }

        // per spray group info
        ioctlParams.portsPerSprayGroup[i] = portList.ports_size();
        if (portList.has_replicavalid()) {
            ioctlParams.replicaValid[i] = portList.replicavalid();
        }
        if (portList.has_replicaoffset()) {
            ioctlParams.replicaOffset[i] = portList.replicaoffset();
        }

        // per spray group port list
        for (int j = 0; j < portList.ports_size(); j++) {

            if ( portIndex >= LWSWITCH_MC_MAX_PORTS ) {
                FM_LOG_ERROR("multicast group index %d port %d is greater than the max number of ports allowed %d.",
                             ioctlParams.index, portIndex, LWSWITCH_MC_MAX_PORTS);
                return false;
            }

            const lwswitch::MulticastPort &portInfo = portList.ports(j);
            ioctlParams.ports[portIndex] = portInfo.portnum();
            ioctlParams.vcHop[portIndex] = portInfo.vchop();

            portIndex++;
        }
    }

    return true;
}

void
LocalFMControlMsgHndl::handleMulticastIdRequestMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    const lwswitch::mcidTableSetRequest &configRequest = pFmMessage->mcidtablesetreq();
    LocalFMSwitchInterface *interface;
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_MC_RID_TABLE_PARAMS ioctlParams;

    memset(&ioctlStruct, 0, sizeof(ioctlStruct));
    ioctlStruct.type = IOCTL_LWSWITCH_SET_MC_RID_TABLE;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    FMIntReturn_t fmResult = FM_INT_ST_OK;
    lwswitch::mcidTableSetResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::mcidTableSetResponse;
    pFmResponse->set_allocated_mcidtablesetrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_MCID_TABLE_SET_RSP );

    lwswitch::configResponse *instanceResponse = pResponse->add_response();

    interface = mpControl->switchInterfaceAt( configRequest.switchphysicalid() );

    if (interface == NULL) {
        FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d",
                      configRequest.switchphysicalid());
        instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
    } else {

        for  (int g = 0; g < configRequest.groupinfo_size(); g++ ) {

            const lwswitch::MulticastGroupInfo &groupInfo = configRequest.groupinfo(g);
            const lwswitch::SwitchMultcastTableInfo &switchTable = groupInfo.switchmulticasttable();

            for  (int i = 0; i < switchTable.portmulticasttable_size(); i++ ) {

                const lwswitch::PortMulticastTableInfo &portInfo = switchTable.portmulticasttable(i);

                if ( portInfo.has_tableentry() ) {
                    if (!mcIdEntryToIoctlParams(portInfo.portnum(), groupInfo.nodynrsp(),
                                                portInfo.tableentry(), ioctlParams) ||
                        (interface->doIoctl( &ioctlStruct ) == FM_INT_ST_OK)) {

                        instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
                    } else {
                        instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
                        break;
                    }
                }

                if ( portInfo.has_extendedtableentry() ) {
                    if (!mcIdEntryToIoctlParams(portInfo.portnum(), groupInfo.nodynrsp(),
                                                portInfo.extendedtableentry(), ioctlParams) ||
                        (interface->doIoctl( &ioctlStruct ) == FM_INT_ST_OK)) {

                        instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
                    } else {
                        instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
                        break;
                    }
                }
            }
        }
    }

    // send the response back to gfm
    fmResult = SendMessageToGfm( pFmResponse, false );
    if ( fmResult != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send multicast config response message");
    }
    delete pFmResponse;
}
#endif

void
LocalFMControlMsgHndl::handleDegradedGpuInfoMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int i;
    lwswitch::gpuDegradedInfoAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuDegradedInfoAck;
    pFmResponse->set_allocated_gpudegradedinfoack( pResponse );
    pFmResponse->set_type( lwswitch::FM_DEGRADED_GPU_INFO_ACK );

    const lwswitch::gpuDegradedInfo &degradedInfo = pFmMessage->gpudegradedinfo();

    for ( i = 0; i < degradedInfo.gpuinfo_size(); i++ )
    {
        const lwswitch::gpuDegraded *pGpuDegraded = &degradedInfo.gpuinfo(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();

        instanceResponse->set_devicephysicalid( pGpuDegraded->physicalid() );

        // TODO, access OOB API to write degraded GPU reason
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("control msg handler: failed to send GPU degraded info config response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::handleDegradedSwitchInfoMsg( lwswitch::fmMessage *pFmMessage )
{
    FMIntReturn_t rc;
    int i;
    lwswitch::switchDegradedInfoAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    FMIntReturn_t fmResult = FM_INT_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::switchDegradedInfoAck;
    pFmResponse->set_allocated_switchdegradedinfoack( pResponse );
    pFmResponse->set_type( lwswitch::FM_DEGRADED_LWSWITCH_INFO_ACK );

    const lwswitch::switchDegradedInfo &degradedInfo = pFmMessage->switchdegradedinfo();

    for ( i = 0; i < degradedInfo.switchinfo_size(); i++ )
    {
        const lwswitch::switchDegraded *pSwitchDegraded = &degradedInfo.switchinfo(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();

        instanceResponse->set_devicephysicalid( pSwitchDegraded->physicalid() );

        LocalFMSwitchInterface *pSwitchIntf;
        pSwitchIntf = mpControl->switchInterfaceAt(pSwitchDegraded->physicalid());

        if (pSwitchIntf == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d", pSwitchDegraded->physicalid());
            instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
            continue;
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_BLACKLIST_DEVICE_PARAMS ioctlParams;

        memset(&ioctlParams, 0, sizeof(ioctlParams));

        LWSWITCH_DEVICE_BLACKLIST_REASON switchDegradeReason;
        colwertFmToDriverDegradedReason(pSwitchDegraded->reason(), switchDegradeReason);

        ioctlParams.deviceReason = switchDegradeReason;

        ioctlStruct.type = IOCTL_LWSWITCH_BLACKLIST_DEVICE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        rc = pSwitchIntf->doIoctl( &ioctlStruct );

        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to set degrade reason for LWSwitch physical id %d", pSwitchDegraded->physicalid());
            instanceResponse->set_status(lwswitch::CONFIG_FAILURE);
        }
        else {
            instanceResponse->set_status(lwswitch::CONFIG_SUCCESS);
            // after this call, do not try to access pSwitchIntf below, it will be NULL
            mpControl->addDegradedSwitchInfo(pSwitchDegraded->physicalid(), pSwitchDegraded->reason());
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_INT_ST_OK )
    {
         FM_LOG_ERROR("control msg handler: failed to send switch degraded info config response message");
    }

    delete pFmResponse;
}

void
LocalFMControlMsgHndl::colwertDriverToFmDegradedReason(lwswitch::SwitchDegradedReason &reason, 
                                                       LWSWITCH_DEVICE_BLACKLIST_REASON excludedReason)
{
    switch (excludedReason) {           
        case LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE:    
            reason = lwswitch::TRUNK_LINK_FAILURE;
            break;

        case LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE:
            reason = lwswitch::ACCESS_LINK_FAILURE;
            break;

        case LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE_PEER:
            reason = lwswitch::PEER_LWSWITCH_DEGRADED_ACCESS_LINK;
            break;

        case LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE_PEER:
            reason = lwswitch::PEER_LWSWITCH_DEGRADED_TRUNK_LINK;
            break;

        case LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_PEER:
            reason = lwswitch::PEER_DEGRADE_EXPLICITLY_EXCLUDED_SWITCH;
            break;

        case LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_IN_BAND:
        case LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND:
            reason = lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED;
            break;

        default:
            reason = lwswitch::LWSWITCH_EXPLICITLY_EXCLUDED;
            break;
    }
}

void
LocalFMControlMsgHndl::colwertFmToDriverDegradedReason(lwswitch::SwitchDegradedReason reason, 
                                                       LWSWITCH_DEVICE_BLACKLIST_REASON &excludedReason)
{
    switch (reason) {
        case lwswitch::LWSWITCH_FAILURE:
        case lwswitch::LWSWITCH_PEER_FAILURE:
            break;
           
        // degrading switch due to trunk link failure
        case lwswitch::TRUNK_LINK_FAILURE:    
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE;
            break;

        // degrading switch due to access link failure
        case lwswitch::ACCESS_LINK_FAILURE:
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE;
            break;

        // degrading switch due to peer switch having access link failure
        case lwswitch::PEER_LWSWITCH_DEGRADED_ACCESS_LINK:
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_ACCESS_LINK_FAILURE_PEER;
            break;

        // degrading switch due to peer switch having access link failure
        case lwswitch::PEER_LWSWITCH_DEGRADED_TRUNK_LINK:
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_TRUNK_LINK_FAILURE_PEER;
            break;

        // degrading switch due to peer switch being manually excluded
        case lwswitch::PEER_DEGRADE_EXPLICITLY_EXCLUDED_SWITCH:
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_PEER;
            break;

        default:
            excludedReason = LWSWITCH_DEVICE_BLACKLIST_REASON_UNSPEC_DEVICE_FAILURE;
            break;
    }
}

/**
 *  on LFM, send message to GFM
 */
FMIntReturn_t
LocalFMControlMsgHndl::SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    FMIntReturn_t ret;

    if ( mpControl == NULL )
    {
        FM_LOG_DEBUG("Invalid control connection to gfm");
        return FM_INT_ST_ILWALID_LOCAL_CONTROL_CONN_TO_GFM;
    }

    ret = mpControl->SendMessageToGfm( pFmMessage, trackReq );
    if ( ret != FM_INT_ST_OK )
    {
        return FM_INT_ST_MSG_SEND_ERR;
    }
    return FM_INT_ST_OK;
}

void
LocalFMControlMsgHndl::colwertFmToDriverRemapPolicyFlags(uint32_t fmFlags,
                                                         uint32_t &driverFlags)
{
    driverFlags = 0;

    if (fmFlags & REMAP_FLAGS_REMAP_ADDR) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR;
    }

    if (fmFlags & REMAP_FLAGS_REQCTXT_CHECK) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK;
    }

    if (fmFlags & REMAP_FLAGS_REQCTXT_REPLACE) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE;
    }

    if (fmFlags & REMAP_FLAGS_ADR_BASE) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE;
    }

    if (fmFlags & REMAP_FLAGS_ADR_OFFSET) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET;
    }

    if (fmFlags & REMAP_FLAGS_REFLECTIVE) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_REFLECTIVE;
    }

    if (fmFlags & REMAP_FLAGS_ADDR_TYPE) {
        driverFlags |= LWSWITCH_REMAP_POLICY_FLAGS_ADDR_TYPE;
    }
}

void
LocalFMControlMsgHndl::gfmHeartbeatTimeoutHandler(void *arg)
{
    // Timer expired
    FM_LOG_ERROR(" heartbeats from GFM stopped for %d seconds", HEARTBEAT_INTERVAL * HEARTBEAT_THRESHOLD);
    LocalFMControlMsgHndl *pObj = (LocalFMControlMsgHndl *)arg;
    pObj->gfmHeartbeatTimeoutProcess();
}

void
LocalFMControlMsgHndl::gfmHeartbeatTimeoutProcess()
{
    mpControl->onConfigDeInitReqRcvd();
}

