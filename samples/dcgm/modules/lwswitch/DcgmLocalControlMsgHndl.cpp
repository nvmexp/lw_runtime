#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include "logging.h"
#include "LwcmHostEngineHandler.h"
#include "DcgmLocalControlMsgHndl.h"

#include "lwml.h"
#include "lwml_internal.h"
#include <g_lwconfig.h>


DcgmLocalControlMsgHndl::DcgmLocalControlMsgHndl(DcgmLocalFabricManagerControl       *pControl,
                                           etblLWMLCommonInternal_st * etblLwmlCommonInternal)
{
    int i, numOfQueues;
    workqueue_t *pWorkqueue;

    mpControl                = pControl;
    metblLwmlCommonInternal  = etblLwmlCommonInternal;

    // create a workqueue for each switch interface, for switch message processing
    numOfQueues = mpControl->getNumLwswitchInterface();

    for ( i = 0; i < numOfQueues; i++ )
    {
        pWorkqueue = new workqueue_t;
        mvWorkqueue.push_back( pWorkqueue );
        if ( workqueue_init(pWorkqueue, 1) ) {
            PRINT_ERROR("", "failed to create control message handler worker queues");
            throw std::runtime_error("failed to create control message handler worker queues");
        }
    }
};

DcgmLocalControlMsgHndl::~DcgmLocalControlMsgHndl()
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
};

void
DcgmLocalControlMsgHndl::handleEvent( FabricManagerCommEventType eventType, uint32 nodeId )
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_PEER_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_PEER_FM_DISCONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_GLOBAL_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            PRINT_INFO("%d", "nodeId %d FM_EVENT_GLOBAL_FM_DISCONNECT", nodeId);
            break;
        }
    }
}

void
DcgmLocalControlMsgHndl::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());
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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
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
#endif
    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        return;
    }

    // for messages which are not handled directly, process them using the
    // associated worker thread
    if (queueNum < 0)
    {
        // we shouldn't land here with an invalid worker queue number
        PRINT_ERROR("", "invalid worker queue number to process config message");
        return;
    }

    job_t *pJob = new job_t;
    workqueue_t *pQueue = NULL;

    // do all the error/validation before allocating messages/job object
    if ( queueNum >= (int)mvWorkqueue.size() )
    {
        PRINT_ERROR("%d", "assigned worker queue number %d is out of range", queueNum);
        return;
    }

    // enqueue the message to be processed by work queue thread    
    pQueue = mvWorkqueue.at(queueNum);
    if ( pQueue == NULL )
    {
        PRINT_ERROR("%d", "Invalid worker queue object for queue number  %d", queueNum);
        return;
    }

    // make a copy of the message, because pFmMessage will be
    // freed after return from this method
    // DcgmLocalControlMsgHndl::processMessage will free the copy
    lwswitch::fmMessage *pMsg = pFmMessage->New();
    pMsg->CopyFrom(*pFmMessage);

    FmMessageUserData_t *pUsrData = new FmMessageUserData_t;
    pJob->user_data = pUsrData;
    pUsrData->pFmMessage = pMsg;
    pUsrData->pHndl = this;

    pJob->job_function = DcgmLocalControlMsgHndl::processMessageJob;

    workqueue_add_job(pQueue, pJob);
}

void
DcgmLocalControlMsgHndl::processMessage( lwswitch::fmMessage *pFmMessage )
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());

    PRINT_DEBUG("%d", "pthread_t %d", (int)pthread_self());

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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    case lwswitch::FM_RMAP_TABLE_REQ:
        handlePortRmapConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_RID_TABLE_REQ:
        handlePortRidConfigReqMsg( pFmMessage );
        break;

    case lwswitch::FM_RLAN_TABLE_REQ:
        handlePortRlanConfigReqMsg( pFmMessage );
        break;
#endif
    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        break;
    }

    delete pFmMessage;
}

/**
 *  thread function to process FM control messages
 */
void
DcgmLocalControlMsgHndl::processMessageJob( job_t *pJob )
{
    FmMessageUserData_t *pUsrData = (FmMessageUserData_t *) pJob->user_data;
    pUsrData->pHndl->processMessage(pUsrData->pFmMessage);

    delete pUsrData;
    pUsrData = NULL;
    delete pJob;
}

void
DcgmLocalControlMsgHndl::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    PRINT_DEBUG("%s", "%s", msgText.c_str());
#endif
}

/**
 *  on LFM, handle FM_NODE_GLOBAL_CONFIG_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleNodeGlobalConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    lwswitch::fmMessage *pFmResponse;
    lwswitch::nodeGlobalConfigResponse *pRespMsg;

    if ( !pFmMessage->has_globalconfigrequest() ) {
        PRINT_WARNING("", "received Node Global Config Request message without required fields\n");
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
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_SWITCH_PORT_CONFIG_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleSWPortConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int ret = 0, i, requestId;
    dcgmReturn_t dcgmReturn;
    DcgmSwitchInterface *interface;
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
            PRINT_ERROR("%d", "unknown switch port type %d",
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        ioctlParams.requesterLanID = config->has_rlanid() ? config->rlanid() : 0;
        ioctlParams.count = config->has_maxtargetid() ? config->maxtargetid() : 0;
#endif

        ioctlStruct.type = IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG;
        ioctlStruct.ioctlParams = &ioctlParams;

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( configRequest.switchphysicalid() );
        if ( !interface )
        {
            PRINT_ERROR("%d", "Invalid switch interface at %d.",
                        configRequest.switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        rc = interface->doIoctl( &ioctlStruct );
        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_INGRESS_REQUEST_TABLE_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleIngReqTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    int ret = 0, i, j, requestId, entryCount;
    DcgmSwitchInterface *interface;
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

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            PRINT_ERROR("%d", "Invalid switch interface at %d.",
                        info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
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
                if ( rc != FM_SUCCESS )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_SUCCESS ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_INGRESS_RESPONSE_TABLE_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleIngRespTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    int ret = 0, i, j, requestId, entryCount;
    DcgmSwitchInterface *interface;
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

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            PRINT_ERROR("%d", "Invalid switch interface at %d.",
                        info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        ioctlStruct.type = IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;
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
                if ( rc != FM_SUCCESS ) {
                    break;
                }
            }
        }

        if ( ( rc == FM_SUCCESS ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GANGED_LINK_TABLE_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleGangedLinkTblConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int ret = 0, i, j, index, requestId;
    DcgmSwitchInterface *interface;
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

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );
        if ( !interface )
        {
            PRINT_ERROR("%d", "Invalid switch interface at %d.",
                        info->switchphysicalid());
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // construct the ioctl on the new port
        ioctlStruct.type = IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE;
        ioctlStruct.ioctlParams = &ioctlParams;

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
        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_CONFIG_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleGpuConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int i;
    lwswitch::gpuConfigResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager

    lwmlDevice_t lwmlDevice;
    lwmlBaseAddress_t baseAddress;
    lwmlReturn_t lwmlResult;

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

        // set up structures and call RM controls to set base fabric address
        // and peer ID to port map based on the request

        // to configure, use the LWML GPU index instead of the physical id
        int lwmlIndex = mpControl->mpCacheManager->GpuIdToLwmlIndex( pInfo->gpuenumindex() );
        if ( lwmlIndex < 0 )
        {
             PRINT_ERROR( "%d", "GpuIdToLwmlIndex failed for gpuId %d.",
                           pInfo->gpuphysicalid() );
             lwmlResult = LWML_ERROR_NOT_FOUND;

        } else {
            lwmlResult = lwmlDeviceGetHandleByIndex( lwmlIndex, &lwmlDevice );
            if(LWML_SUCCESS != lwmlResult)
            {
                PRINT_ERROR("%d %d %d",
                            "lwmlDeviceGetHandleByIndex failed for GPU PhysicalId %d EnumIndex %d with %d",
                            pInfo->gpuphysicalid(), lwmlIndex, lwmlResult);
            } else {

                if ( pInfo->has_fabricaddressbase() )
                {
                    baseAddress.address =pInfo->fabricaddressbase();
                    PRINT_INFO("%d %d %llx",
                                "Setting GPU Base address for PhysicalId %d EnumIndex %d Base Address %llx",
                                pInfo->gpuphysicalid(), pInfo->gpuenumindex(), baseAddress.address);

                    lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal,
                                                 DeviceSetBaseAddress,
                                                 (lwmlDevice, &baseAddress) );
                    if(LWML_SUCCESS != lwmlResult)
                    {
                        PRINT_ERROR("%d %d %d",
                                    "lwmlDeviceSetBaseAddress failed for GPU PhysicalId %d EnumIndex %d with %d",
                                    pInfo->gpuphysicalid(), lwmlIndex, lwmlResult);
                    }
                }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
                if ( pInfo->has_gpaaddressbase() && pInfo->has_gpaaddressrange() )
                {
                    baseAddress.address =pInfo->gpaaddressbase();
                    PRINT_INFO("%d %d %llx",
                                "Setting GPU GPA Base address for PhysicalId %d EnumIndex %d Base Address %llx",
                                pInfo->gpuphysicalid(), pInfo->gpuenumindex(), baseAddress.address);

                    lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal,
                                                 DeviceSetBaseAddress,
                                                 (lwmlDevice, &baseAddress) );
                    if(LWML_SUCCESS != lwmlResult)
                    {
                        PRINT_ERROR("%d %d %d",
                                    "lwmlDeviceSetBaseAddress failed for GPU PhysicalId %d EnumIndex %d with %d",
                                    pInfo->gpuphysicalid(), lwmlIndex, lwmlResult);
                    }
                }

                if ( pInfo->has_flaaddressbase() && pInfo->has_flaaddressrange() )
                {
                    lwmlFlaRange_t flaAddress;
                    int mode = 0; // LW2080_CTRL_FLA_RANGE_PARAMS_MODE_INITIALIZ
                    flaAddress.address =pInfo->flaaddressbase();
                    flaAddress.size = pInfo->flaaddressrange();

                    PRINT_INFO("%d %d %llx %llx",
                                "Setting GPU FLA Base address for PhysicalId %d EnumIndex %d Base Address %llx Range %llx ",
                                pInfo->gpuphysicalid(), pInfo->gpuenumindex(), flaAddress.address, flaAddress.size);

                    lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal,
                                                 DeviceSetFlaRange,
                                                 (lwmlDevice, &flaAddress, mode) );
                    if(LWML_SUCCESS != lwmlResult)
                    {
                        PRINT_ERROR("%d %d %d",
                                    "lwmlDeviceSetFlaAddress failed for GPU PhysicalId %d EnumIndex %d with %d",
                                    pInfo->gpuphysicalid(), lwmlIndex, lwmlResult);
                    }
                }
#endif
            }
        }
        
        if ( lwmlResult == LWML_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_ATTACH_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleGpuAttachReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int i;
    lwswitch::gpuAttachResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    dcgmReturn_t dcgmResult = DCGM_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuAttachResponse;
    pFmResponse->set_allocated_gpuattachrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_GPU_ATTACH_RSP );

    const lwswitch::gpuAttachRequest &configRequest = pFmMessage->gpuattachreq();

    for ( i = 0; i < configRequest.info_size(); i++ )
    {
        const lwswitch::gpuInfo * pInfo = &configRequest.info(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( pInfo->gpuphysicalid() );
        if ( configRequest.has_partitionid() )
            instanceResponse->set_partitionid( configRequest.partitionid() );
        else
            instanceResponse->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

        // call DCGM api to attach GPU
        // this is only done once, the loop is preserved
        // just in case the future lwml attach API is per GPU based.
        dcgmResult = mpControl->attachAllGpus();
        if ( dcgmResult == DCGM_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }

        // All GPUs are attached together
        break;
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_GPU_DETACH_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleGpuDetachReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int i;
    lwswitch::gpuDetachResponse *pResponse;
    lwswitch::fmMessage *pFmResponse;  //response to send back to Global Fabric Manager
    dcgmReturn_t dcgmResult = DCGM_ST_OK;

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::gpuDetachResponse;
    pFmResponse->set_allocated_gpudetachrsp( pResponse );
    pFmResponse->set_type( lwswitch::FM_GPU_DETACH_RSP );

    const lwswitch::gpuDetachRequest &configRequest = pFmMessage->gpudetachreq();

    for ( i = 0; i < configRequest.info_size(); i++ )
    {
        const lwswitch::gpuInfo * pInfo = &configRequest.info(i);

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( pInfo->gpuphysicalid() );
        if ( configRequest.has_partitionid() )
            instanceResponse->set_partitionid( configRequest.partitionid() );
        else
            instanceResponse->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

        // call DCGM api to detach all GPU
        // this is only done once, the loop is preserved
        // just in case the future lwml detach API is per GPU based.
        dcgmResult = mpControl->detachAllGpus();
        if ( dcgmResult == DCGM_ST_OK ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }

        // All GPUs are detached together
        break;
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_CONFIG_INIT_DONE_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleConfigInitDoneReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
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
    if ( rc != FM_SUCCESS ) {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_CONFIG_DEINIT_REQ message from GFM
 */
void
DcgmLocalControlMsgHndl::handleConfigDeInitReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
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
    if ( rc != FM_SUCCESS ) {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_HEARTBEAT message
 */
void
DcgmLocalControlMsgHndl::handleHeartbeatMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    int i;
    const lwswitch::heartbeat *pRequest = &(pFmMessage->heartbeat());
    lwswitch::heartbeatAck *pResponse;
    lwswitch::fmMessage *pFmResponse;  //heartbeat ack to send back to GFM

    pFmResponse = new lwswitch::fmMessage();
    pFmResponse->set_type( lwswitch::FM_HEARTBEAT_ACK );
    pFmResponse->set_requestid( pFmMessage->requestid() );

    pResponse = new lwswitch::heartbeatAck;
    pFmResponse->set_allocated_heartbeatack( pResponse );

    if ( mpControl )
    {
        // send the heartbeat ack to gfm
        mpControl->SendMessageToGfm( pFmResponse, false );
    }

    delete pFmResponse;
}

/**
 *  on LFM, handle FM_SWITCH_DISABLE_LINK_REQ message
 */
void
DcgmLocalControlMsgHndl::handleSwitchDisableLinkReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    DcgmSwitchInterface *interface = NULL;
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
            rc = interface->doIoctl( &ioctlStruct );
            if ( rc != FM_SUCCESS )
            {
                PRINT_ERROR( "%d %d", "Disable link failed for LWSwitch ID %d Port ID %d",
                            disableReqMsg.switchphysicalid(), portNum);
                retStatus = lwswitch::CONFIG_FAILURE;
                break;
            }
        }
        // update enabled port mask information after links are disabled
        interface->updateEnabledPortMask();
    } else
    {
        PRINT_ERROR( "%d", "Invalid switch interface at %d.", disableReqMsg.switchphysicalid() );
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
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG( " ", "Failed to send response to gfm" );
    }

    delete pFmResponse;
}

void
DcgmLocalControlMsgHndl::handleGpuSetDisabledLinkMaskReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc;
    lwmlReturn_t lwmlResult = LWML_SUCCESS;
    lwswitch::configStatus retStatus = lwswitch::CONFIG_SUCCESS;
    lwswitch::gpuSetDisabledLinkMaskResponse *pMaskRespMsg = new lwswitch::gpuSetDisabledLinkMaskResponse();

    //
    // GPU LWLink disabled mask should be written before the GPU is initialized/attached by any LWML client.
    // For this, we need to first initialize LWML with the LWML_INIT_FLAG_NO_ATTACH and set the mask using
    // the internal API. This lwmlInitWithFlags() and lwmlShutdown() sequence will go away once LWML support
    // dynamic GPU enumeration/attach/detach sequence in eGPU time frame.
    //

    //
    // Note: This is very specific to Shared Fabric Mode, which assumes that the GPUs are not attached by
    // any LWML client once they are hot plugged into the ServiceVM as part of partition activation.
    //

    // initialize LWML with NO_ATTACH to populate/enumerate the hot plugged GPUs.
    lwmlInitWithFlags( LWML_INIT_FLAG_NO_ATTACH );

    // set disabled link mask for each GPU
    const lwswitch::gpuSetDisabledLinkMaskRequest &disableMaskReqMsg = pFmMessage->gpusetdisabledlinkmaskreq();

    for ( int idx = 0; idx < disableMaskReqMsg.gpuinfo_size(); idx++ )
    {
        const lwswitch::gpuDisabledLinkMaskInfoMsg &gpuInfoMsg = disableMaskReqMsg.gpuinfo( idx );
        // we must have gpu uuid to set the disabled link mask
        if ( !gpuInfoMsg.has_uuid() )
        {
            PRINT_ERROR(" ", "handleGpuSetDisabledLinkMaskReqMsg: Missing GPU uuid information");
            retStatus = lwswitch::CONFIG_FAILURE;
            break;
        }
        const char *uuid = gpuInfoMsg.uuid().c_str();
        uint32 disabledMask = gpuInfoMsg.disablemask();
        lwmlResult = LWML_CALL_ETBL( metblLwmlCommonInternal, 
                                     DeviceDisableLwlinkInit, (uuid, disabledMask) );

        if ( LWML_SUCCESS != lwmlResult )
        {
            PRINT_ERROR("%s %d", "DeviceDisableLwlinkInit failed for GPU UUID %s with error %d",
                        uuid, lwmlResult);
            // report the failed GPU information and mark the request as failed.
            pMaskRespMsg->set_uuid( uuid );
            retStatus = lwswitch::CONFIG_FAILURE;
            break;
        }
    }

    // 
    // Done with all the GPU mask settings. Initiate NMVL shutdown to clean the lib context.
    // Later as a continuation of partition activation, DCGM will attach the GPUs normally
    // and FM will initiate link training and further configuration.
    // 
    lwmlShutdown();

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
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG( " ", "Failed to send response to gfm" );
    }

    delete pFmResponse;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
DcgmLocalControlMsgHndl::handlePortRmapConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    int ret = 0, i, j, requestId, entryCount;
    DcgmSwitchInterface *interface;
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

        lwswitch::configResponse *instanceResponse = pResponse->add_response();
        instanceResponse->set_devicephysicalid( info->switchphysicalid() );
        instanceResponse->set_port( info->port() );

        // send the partitionId back in the response
        if ( info->has_partitionid() )
            instanceResponse->set_partitionid( info->partitionid() );

        entryCount = 0;

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        ioctlStruct.type = IOCTL_LWSWITCH_SET_REMAP_POLICY;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlParams.portNum = info->port();

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
                if ( rc != FM_SUCCESS )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_SUCCESS ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

void
DcgmLocalControlMsgHndl::handlePortRidConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    int ret = 0, i, j, p, requestId, entryCount;
    DcgmSwitchInterface *interface;
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

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        ioctlStruct.type = IOCTL_LWSWITCH_SET_ROUTING_ID;
        ioctlStruct.ioctlParams = &ioctlParams;
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
                if ( rc != FM_SUCCESS )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_SUCCESS ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

void
DcgmLocalControlMsgHndl::handlePortRlanConfigReqMsg( lwswitch::fmMessage *pFmMessage )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    int ret = 0, i, j, p, requestId, entryCount;
    DcgmSwitchInterface *interface;
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

        if ( !mpControl )
        {
            PRINT_ERROR(" ", "Invalid Local Control." );
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
            continue;
        }

        // TODO should use switch index here
        interface = mpControl->switchInterfaceAt( info->switchphysicalid() );

        ioctlStruct.type = IOCTL_LWSWITCH_SET_ROUTING_LAN;
        ioctlStruct.ioctlParams = &ioctlParams;
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
            memset(&ioctlParams.routingLan[entryCount], 0, sizeof(LWSWITCH_ROUTING_ID_ENTRY));

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
                if ( rc != FM_SUCCESS )
                {
                    break;
                }
            }
        }

        if ( ( rc == FM_SUCCESS ) && ( entryCount > 0 ) )
        {
            // issue the ioctl for the remaining entries
            ioctlParams.numEntries = entryCount;
            entryCount = 0;
            rc = interface->doIoctl( &ioctlStruct );
        }

        if ( rc == FM_SUCCESS ) {
            instanceResponse->set_status( lwswitch::CONFIG_SUCCESS );
        } else {
            instanceResponse->set_status( lwswitch::CONFIG_FAILURE );
        }
    }

    // send the response back to gfm
    rc = SendMessageToGfm( pFmResponse, false );
    if ( rc != FM_SUCCESS )
    {
        PRINT_DEBUG(" ", "Failed to send response to gfm");
    }

    delete pFmResponse;
}

#endif

/**
 *  on LFM, send message to GFM
 */
FM_ERROR_CODE
DcgmLocalControlMsgHndl::SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    dcgmReturn_t ret;

    if ( mpControl == NULL )
    {
        PRINT_DEBUG("", "Invalid control connection to gfm");
        return FM_ILWALID_LOCAL_CONTROL_CONN_TO_GFM;
    }

    ret = mpControl->SendMessageToGfm( pFmMessage, trackReq );
    if ( ret != DCGM_ST_OK )
    {
        return FM_MSG_SEND_ERR;
    }
    return FM_SUCCESS;
}

