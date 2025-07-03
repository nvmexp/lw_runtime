
#include <stdlib.h>
#include "logging.h"
#include "DcgmFMLWLinkIntf.h"
#include "DcgmFMAutoLock.h"
#include <g_lwconfig.h>



DcgmFMLWLinkIntf::DcgmFMLWLinkIntf(FMConnInterface *ctrlConnIntf)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf::DcgmFMLWLinkIntf\n" );

    lwosInitializeCriticalSection( &mLock );
    mCtrlConnIntf = ctrlConnIntf;
    mNextReqId = 0; //valid IDs start from 1
    mTrainReqPending.clear();
    mTrainReqComplete.clear();
}

DcgmFMLWLinkIntf::~DcgmFMLWLinkIntf()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf::DcgmFMLWLinkIntf\n" );
    // since the map has stack objects, just empty the map container
    mTrainReqPending.clear();
    mTrainReqComplete.clear();
    lwosDeleteCriticalSection( &mLock );
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendEnableCommonModeReq(DcgmLWLinkReq &linkReq,
                                          uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendDisableCommonModeReq(DcgmLWLinkReq &linkReq,
                                           uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendCalibrateReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_CALIBRATE, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendEnableDataReq(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_ENABLE_DATA, linkReq, requestId);
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
dcgmReturn_t
DcgmFMLWLinkIntf::sendInitphase1Req(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_INITPHASE1, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendRxInitTermReq(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_RX_INIT_TERM, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendSetRxDetectReq(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_SET_RX_DETECT, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendGetRxDetectReq(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_GET_RX_DETECT, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendInitnegotiateReq(DcgmLWLinkReq &linkReq,
                                    uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_INITNEGOTIATE, linkReq, requestId);
}
#endif

dcgmReturn_t
DcgmFMLWLinkIntf::sendLinkInitReq(DcgmLWLinkReq &linkReq,
                                  uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_INIT, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendLinkInitStatusReq(DcgmLWLinkReq &linkReq,
                                        uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_INIT_STATUS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendResetSwitchLinksReq(DcgmLWLinkReq &linkReq,
                                          uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_RESET_SWITCH_LINKS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendResetAllSwitchLinksReq(DcgmLWLinkReq &linkReq,
                                             uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendConnTrainReq(DcgmLWLinkReq &linkReq,
                                   uint64 &requestId)
{
    switch( linkReq.connTrainReq.trainTo ) {
        case LWLINK_TRAIN_OFF_TO_SAFE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_HIGH: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH, linkReq, requestId);
        }
        case LWLINK_TRAIN_TO_OFF: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF, linkReq, requestId);
        }
        case LWLINK_TRAIN_HIGH_TO_SAFE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_OFF: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF, linkReq, requestId);
        }
    } 

    return DCGM_ST_BADPARAM;
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendDiscoverIntraNodeConnReq(DcgmLWLinkReq &linkReq,
                                               uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendAddInterNodeConnReq(DcgmLWLinkReq &linkReq,
                                          uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_ADD_INTERNODE_CONN, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendGetIntraNodeConnReq(DcgmLWLinkReq &linkReq,
                                          uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_GET_INTRANODE_CONNS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendWriteDiscoveryReq(DcgmLWLinkReq &linkReq,
                                        uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS, linkReq, requestId);
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendReadDiscoveryReq(DcgmLWLinkReq &linkReq,
                                       uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS, linkReq, requestId);
}

bool
DcgmFMLWLinkIntf::isLinkReqComplete(uint64 requestId,
                                    DcgmLWLinkReqResult &reqResult)
{
    DcgmFMAutoLock lock(mLock);
    TrainRequestMap::iterator it;

    it = mTrainReqComplete.find( requestId );
    if ( it != mTrainReqComplete.end() ) {
        DcgmLWLinkReqCtx reqCtx = it->second;
        reqResult = reqCtx.result;
        mTrainReqComplete.erase( requestId );
        return true;
    }
    return false;
}

void
DcgmFMLWLinkIntf::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE: {
            handleTrainReqResponseMsg( pFmMessage );
            break;
        }
        default: {
            PRINT_WARNING( "", "Link training interface received unknown message type\n" );
            break;
        }
    }
}

void
DcgmFMLWLinkIntf::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf handleEvent\n" );

    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // do nothing
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            handleNodeDisconnect( nodeId );
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // trainings are initiated from GFM. So not applicable for GFM itself
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // trainings are initiated from GFM. So not applicable for GFM itself
            break;
        }
    }
}

void
DcgmFMLWLinkIntf::dumpInfo(std::ostream *os)
{
    DcgmFMAutoLock lock(mLock);

    *os << "\t\tDcgmFMLWLinkIntf Dump Start" << std::endl;

    TrainRequestMap::iterator it = mTrainReqPending.begin();

    *os << "\t\tDumping pending request info" << std::endl;
    while ( it != mTrainReqPending.end() ) {
        dumpTrainCtxEntry( os, it->first, it->second );
        ++it;
    }

    *os << "\t\tDumping completed request info" << std::endl;
    it = mTrainReqComplete.begin();
    while ( it != mTrainReqComplete.end() ) {
        dumpTrainCtxEntry( os, it->first, it->second );
        ++it;
    }
    *os << "\t\tDcgmFMLWLinkIntf Dump End" << std::endl;
}

dcgmReturn_t
DcgmFMLWLinkIntf::sendTrainRequest(lwswitch::FabricManagerMessageType msgType,
                                   DcgmLWLinkReq &linkReq,
                                   uint64 &requestId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf sendTrainRequest\n" );
    uint64 reqId = 0;
    dcgmReturn_t retVal;
    uint32 toNodeId = 0;

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkRequestMsg *reqMsg = new lwswitch::lwlinkRequestMsg();

    // fill the lwlinkRequestMsg based on the request type
    switch( msgType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            toNodeId = linkReq.connTrainReq.masterNodeId;            
            genConnTrainReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#endif
        case lwswitch::FM_LWLINK_INIT:{            
            toNodeId = linkReq.nodeInitReq.nodeId;
            genNodeInitReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_INIT_STATUS:{
            toNodeId = linkReq.nodeInitStatusReq.nodeId;
            genNodeInitStatusReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:{
            toNodeId = linkReq.nodeInitResetSwitchLinksReq.nodeId;
            genNodeInitResetSwitchLinksReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS:{            
            toNodeId = linkReq.nodeInitResetAllSwitchLinksReq.nodeId;
            genNodeInitResetAllSwitchLinksReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:{
            toNodeId = linkReq.discoverIntraNodeConnReq.nodeId;            
            genDiscoverIntraNodeConnReqMsg( reqMsg, linkReq);
            break;
        }
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            toNodeId = linkReq.addInterNodeConnReq.localEndInfo.nodeId; 
            genAddInterNodeConnReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            toNodeId = linkReq.getIntraNodeConnReq.nodeId; 
            genGetIntraNodeConnReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:{
            toNodeId = linkReq.writeDiscoveryTokenReq.nodeId; 
            genWriteDiscoveryReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:{
            toNodeId = linkReq.readDiscoveryTokenReq.nodeId; 
            genReadDiscoveryReqMsg( reqMsg, linkReq );
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Link request type while parsing response message\n" );
            break;
        }
    }

    // create the final train request message
    reqId = getNextTrainReqId();
    lwswitch::lwlinkMsg *linkMsg = new lwswitch::lwlinkMsg();
    linkMsg->set_trainreqid( reqId );
    linkMsg->set_allocated_reqmsg( reqMsg );

    // fill the fabric message
    pFmMessage->set_type( msgType );
    pFmMessage->set_allocated_lwlinkmsg( linkMsg );

    // add request to our context for tracking
    // before sending the message, add it to the list as the response can 
    // come before even we add it to the list.
    addToReqPendingTable( reqId, linkReq, msgType, toNodeId );

    // send the message to the master FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( toNodeId, pFmMessage, true );

    // add request to our context for tracking
    if ( retVal != DCGM_ST_OK ) {
        // failed to send the message. remove the request from our local tracking
        removeFromReqPendingTable( reqId );
    } else {
        // send the request successfully.
        // update request id for caller to track
        requestId = reqId;
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
    return retVal;
}

void
DcgmFMLWLinkIntf::genConnTrainReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                     DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkEndPointInfo *masterEnd = new lwswitch::lwlinkEndPointInfo();
    lwswitch::lwlinkEndPointInfo *slaveEnd = new lwswitch::lwlinkEndPointInfo();
    masterEnd->set_nodeid( linkReq.connTrainReq.masterNodeId );
    masterEnd->set_gpuorswitchid( linkReq.connTrainReq.masterGpuOrSwitchId );
    masterEnd->set_linkindex( linkReq.connTrainReq.masterLinkIndex );
    slaveEnd->set_nodeid( linkReq.connTrainReq.slaveNodeId );
    slaveEnd->set_gpuorswitchid( linkReq.connTrainReq.slaveGpuOrSwitchId );
    slaveEnd->set_linkindex( linkReq.connTrainReq.slaveLinkIndex );

    // create the link connection pair
    lwswitch::lwlinkConnectionInfo *connInfo = new lwswitch::lwlinkConnectionInfo();
    connInfo->set_allocated_masterend( masterEnd );
    connInfo->set_allocated_slaveend( slaveEnd );

    lwswitch::lwlinkTrainConnReqMsg *trainConnReqMsg = new lwswitch::lwlinkTrainConnReqMsg();
    trainConnReqMsg->set_allocated_conninfo( connInfo );
    reqMsg->set_allocated_conntrainreqmsg( trainConnReqMsg );
}

void
DcgmFMLWLinkIntf::genNodeInitReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                    DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitReqMsg *nodeInitReqMsg = new lwswitch::lwlinkNodeInitReqMsg();
    reqMsg->set_allocated_nodeinitreqmsg( nodeInitReqMsg );
}

void
DcgmFMLWLinkIntf::genNodeInitStatusReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                          DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitStatusReqMsg *statusReqMsg = new lwswitch::lwlinkNodeInitStatusReqMsg();
    reqMsg->set_allocated_nodeinitstatusreqmsg( statusReqMsg );
}

void
DcgmFMLWLinkIntf::genNodeInitResetSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                    DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg *resetReqMsg = new lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg();
    resetReqMsg->set_switchphysicalid( linkReq.nodeInitResetSwitchLinksReq.switchId );
    resetReqMsg->set_linkmask( linkReq.nodeInitResetSwitchLinksReq.linkMask );
    reqMsg->set_allocated_nodeinitresetswitchlinksreqmsg( resetReqMsg );
}

void
DcgmFMLWLinkIntf::genNodeInitResetAllSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                       DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitResetAllSwitchLinksReqMsg *resetSwitchesReqMsg = new lwswitch::lwlinkNodeInitResetAllSwitchLinksReqMsg();
    reqMsg->set_allocated_nodeinitresetallswitchlinksreqmsg( resetSwitchesReqMsg );
}

void
DcgmFMLWLinkIntf::genDiscoverIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                 DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkDiscoverIntraNodeConnReqMsg *discoverConnReqMsg = new lwswitch::lwlinkDiscoverIntraNodeConnReqMsg();
    reqMsg->set_allocated_discoverintranodeconnreqmsg( discoverConnReqMsg );
}

void
DcgmFMLWLinkIntf::genAddInterNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                            DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkAddInterNodeConnReqMsg *addConnMsg = new lwswitch::lwlinkAddInterNodeConnReqMsg();
    lwswitch::lwlinkInterNodeConnInfo *interNodeConnMsg = new lwswitch::lwlinkInterNodeConnInfo();
    lwswitch::lwlinkEndPointInfo *localEndPoint = new lwswitch::lwlinkEndPointInfo();
    lwswitch::lwlinkRemoteEndPointInfo *remoteEndPoint = new lwswitch::lwlinkRemoteEndPointInfo();
    lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();

    // copy local endpoint information
    localEndPoint->set_nodeid( linkReq.addInterNodeConnReq.localEndInfo.nodeId );
    localEndPoint->set_linkindex( linkReq.addInterNodeConnReq.localEndInfo.linkIndex );
    localEndPoint->set_gpuorswitchid( linkReq.addInterNodeConnReq.localEndInfo.gpuOrSwitchId );

    // copy remote endpoint information
    remoteEndPoint->set_nodeid( linkReq.addInterNodeConnReq.remoteEndInfo.nodeId );
    remoteEndPoint->set_linkindex( linkReq.addInterNodeConnReq.remoteEndInfo.linkIndex );
    pciInfo->set_domain( linkReq.addInterNodeConnReq.remoteEndInfo.pciDomain );
    pciInfo->set_bus( linkReq.addInterNodeConnReq.remoteEndInfo.pciBus );
    pciInfo->set_device( linkReq.addInterNodeConnReq.remoteEndInfo.pciDevice );
    pciInfo->set_function( linkReq.addInterNodeConnReq.remoteEndInfo.pciFunction );
    remoteEndPoint->set_allocated_pciinfo( pciInfo );
    remoteEndPoint->set_devtype( linkReq.addInterNodeConnReq.remoteEndInfo.devType );
    remoteEndPoint->set_uuid( (char*)linkReq.addInterNodeConnReq.remoteEndInfo.uuid );

    interNodeConnMsg->set_allocated_localend( localEndPoint );
    interNodeConnMsg->set_allocated_remoteend( remoteEndPoint );
    addConnMsg->set_allocated_conninfo( interNodeConnMsg );

    reqMsg->set_allocated_addinternodeconnreqmsg( addConnMsg );
}

void
DcgmFMLWLinkIntf::genGetIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                            DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkGetIntraNodeConnReqMsg *getConnReqMsg = new lwswitch::lwlinkGetIntraNodeConnReqMsg();
    reqMsg->set_allocated_getintranodeconnreqmsg( getConnReqMsg );
}

void
DcgmFMLWLinkIntf::genWriteDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                          DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkWriteDiscoveryTokenReqMsg *writeDiscReqMsg = new lwswitch::lwlinkWriteDiscoveryTokenReqMsg();
    reqMsg->set_allocated_writedisctokenreqmsg( writeDiscReqMsg );
}

void
DcgmFMLWLinkIntf::genReadDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                         DcgmLWLinkReq &linkReq)
{
    lwswitch::lwlinkReadDiscoveryTokenReqMsg *readDiscReqMsg = new lwswitch::lwlinkReadDiscoveryTokenReqMsg();
    reqMsg->set_allocated_readdisctokenreqmsg( readDiscReqMsg );
}

uint64
DcgmFMLWLinkIntf::getNextTrainReqId(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf getNextTrainReqId\n" );
    DcgmFMAutoLock lock(mLock);

    mNextReqId++;
    if ( mNextReqId == 0 ) {
        // wrap around
        mNextReqId = 1;
    }
    return mNextReqId;
}

void
DcgmFMLWLinkIntf::addToReqPendingTable(uint64 reqId,
                                       DcgmLWLinkReq &linkReq,
                                       lwswitch::FabricManagerMessageType reqType,
                                       uint32 toNodeId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf addToReqPendingTable\n" );
    DcgmFMAutoLock lock(mLock);

    DcgmLWLinkReqCtx reqCtx = {0};
    reqCtx.reqType = reqType;
    reqCtx.toNodeId = toNodeId;
    reqCtx.req = linkReq;

    if ( mTrainReqPending.count(reqId) ) {
        PRINT_WARNING( "%llu", "Link train request with Train Request ID %llu already present\n", reqId );
        return;
    }

    mTrainReqPending.insert( std::make_pair(reqId, reqCtx) );
}

void
DcgmFMLWLinkIntf::removeFromReqPendingTable(uint64 reqId)
{
    DcgmFMAutoLock lock(mLock);

    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it != mTrainReqPending.end() ) {
        mTrainReqPending.erase( it );
    }
}

bool
DcgmFMLWLinkIntf::getPendingTrainReq(uint64 reqId, DcgmLWLinkReqCtx &reqCtx)
{
    DcgmFMAutoLock lock(mLock);

    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it == mTrainReqPending.end() ) {
        return false;
    }

    // found the request, copy it and return success
    reqCtx = it->second;
    return true;
}

void
DcgmFMLWLinkIntf::markPendingReqAsComplete(uint64 reqId, DcgmLWLinkReqCtx &reqCtx)
{
    DcgmFMAutoLock lock(mLock);
    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it == mTrainReqPending.end() ) {
        return;
    }

    // erase from pending and add to complete list
    mTrainReqPending.erase( it );
    mTrainReqComplete.insert( std::make_pair(reqId, reqCtx) );
}

void
DcgmFMLWLinkIntf::handleTrainReqResponseMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkIntf handleTrainReqResponseMsg\n" );

    // find the corresponding link training request_id and save the result
    // also move the request to the completed requests table

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_WARNING("", "received link training complete message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    if ( !linkMsg.has_rspmsg() ) {
        PRINT_WARNING( "", "received link training complete message without the response message\n" );
        return;
    }

    uint64 reqId = linkMsg.trainreqid();
    DcgmLWLinkReqCtx reqCtx = {0};
    if ( !getPendingTrainReq(reqId, reqCtx) ) {
        PRINT_WARNING( "%llu", "No link train request with trainReqId %llu found during resp handling\n", reqId );
        return;
    }

    // update our local context with result information
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();

    reqCtx.result.requestId = reqId;
    reqCtx.result.status = (DcgmLWLinkErrorCodes)rspMsg.status();

    switch(reqCtx.reqType) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            parseConnTrainResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#endif
        case lwswitch::FM_LWLINK_INIT:
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:    
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS: {
            // nothing specific to copy other than the over-all request status.
            break;
        }
        case lwswitch::FM_LWLINK_INIT_STATUS: {
            parseLinkInitStatusResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            parseGetIntraNodeConnResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:{
            parseWriteDiscoveryTokenResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS:{
            parseReadDiscoveryTokenResp(rspMsg, reqCtx);
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Link request type while parsing response message\n" );
            break;
        }
    }

    // remove from outstanding request and add to completed req table
    markPendingReqAsComplete(reqId, reqCtx);
}

void
DcgmFMLWLinkIntf::parseConnTrainResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                     DcgmLWLinkReqCtx &reqCtx)
{
    lwswitch::lwlinkTrainConnRspMsg connRspMsg = rspMsg.conntrainrspmsg();

    reqCtx.result.connTrainResp.masterState.linkMode = connRspMsg.masterstate().linkmode();
    reqCtx.result.connTrainResp.masterState.txSubLinkMode = connRspMsg.masterstate().txsublinkmode();
    reqCtx.result.connTrainResp.masterState.rxSubLinkMode = connRspMsg.masterstate().rxsublinkmode();
    reqCtx.result.connTrainResp.slaveState.linkMode = connRspMsg.slavestate().linkmode();
    reqCtx.result.connTrainResp.slaveState.txSubLinkMode = connRspMsg.slavestate().txsublinkmode();
    reqCtx.result.connTrainResp.slaveState.rxSubLinkMode = connRspMsg.slavestate().rxsublinkmode();
}

void
DcgmFMLWLinkIntf::parseLinkInitStatusResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                          DcgmLWLinkReqCtx &reqCtx)
{
    lwswitch::lwlinkNodeInitStatusRspMsg statusRspMsg = rspMsg.nodeinitstatusrspmsg();

    // parse status for each device and its links.
    for ( int idx = 0; idx < statusRspMsg.initstatus_size(); idx++ ) {
        lwswitch::lwlinkDeviceLinkInitStatus devStatus = statusRspMsg.initstatus( idx );
        DcgmLinkInitStatusInfo dcgmStatusInfo;
        dcgmStatusInfo.nodeId = statusRspMsg.nodeid();
        dcgmStatusInfo.gpuOrSwitchId = devStatus.gpuorswitchid();
        for ( int linkIdx = 0; linkIdx < devStatus.linkstatus_size(); linkIdx++ ) {
            lwswitch::lwlinkLinkInitStatus linkStatus = devStatus.linkstatus( linkIdx );
            dcgmStatusInfo.initStatus[linkIdx].linkIndex = linkStatus.linkindex();
            dcgmStatusInfo.initStatus[linkIdx].initStatus = linkStatus.status();            
         }
        reqCtx.result.nodeInitStatusResp.statusInfo.push_back( dcgmStatusInfo );
    }
}

void
DcgmFMLWLinkIntf::parseGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                            DcgmLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkGetIntraNodeConnRspMsg getRspMsg = rspMsg.getintranodeconnrspmsg();

    for (idx = 0; idx < getRspMsg.conninfo_size(); idx++) {

        lwswitch::lwlinkConnectionInfo connInfoMsg = getRspMsg.conninfo(idx);
        lwswitch::lwlinkEndPointInfo masterEnd = connInfoMsg.masterend();
        lwswitch::lwlinkEndPointInfo slaveEnd = connInfoMsg.slaveend();

        // parase the google protobuf msg and move to plan format
        DcgmLWLinkConnInfo dcgmConnInfo;
        dcgmConnInfo.masterEnd.nodeId = masterEnd.nodeid();
        dcgmConnInfo.masterEnd.linkIndex = masterEnd.linkindex();
        dcgmConnInfo.masterEnd.gpuOrSwitchId = masterEnd.gpuorswitchid();

        dcgmConnInfo.slaveEnd.nodeId = slaveEnd.nodeid();
        dcgmConnInfo.slaveEnd.linkIndex = slaveEnd.linkindex();
        dcgmConnInfo.slaveEnd.gpuOrSwitchId = slaveEnd.gpuorswitchid();

        reqCtx.result.getIntraNodeConnResp.connInfo.push_back( dcgmConnInfo );
    }
}

void
DcgmFMLWLinkIntf::parseWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                               DcgmLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkWriteDiscoveryTokenRspMsg discRspMsg = rspMsg.writedisctokenrspmsg();

    for (idx = 0; idx < discRspMsg.tokeninfo_size(); idx++) {

        lwswitch::lwlinkDiscoveryTokenInfo tokenInfoMsg = discRspMsg.tokeninfo(idx);

        // parase the google protobuf msg and move to plan format
        DcgmLinkDiscoveryTokenInfo dcgmTokenInfo;
        dcgmTokenInfo.tokelwalue = tokenInfoMsg.tokelwalue();
        dcgmTokenInfo.nodeId = tokenInfoMsg.nodeid();
        dcgmTokenInfo.gpuOrSwitchId = tokenInfoMsg.gpuorswitchid();
        dcgmTokenInfo.linkIndex = tokenInfoMsg.linkindex();

        reqCtx.result.writeDiscoveryTokenResp.tokenInfo.push_back( dcgmTokenInfo );
    }
}

void
DcgmFMLWLinkIntf::parseReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                              DcgmLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkReadDiscoveryTokenRspMsg discRspMsg = rspMsg.readdisctokenrspmsg();

    for (idx = 0; idx < discRspMsg.tokeninfo_size(); idx++) {

        lwswitch::lwlinkDiscoveryTokenInfo tokenInfoMsg = discRspMsg.tokeninfo(idx);

        // parase the google protobuf msg and move to plan format
        DcgmLinkDiscoveryTokenInfo dcgmTokenInfo;
        dcgmTokenInfo.tokelwalue = tokenInfoMsg.tokelwalue();
        dcgmTokenInfo.nodeId = tokenInfoMsg.nodeid();
        dcgmTokenInfo.gpuOrSwitchId = tokenInfoMsg.gpuorswitchid();
        dcgmTokenInfo.linkIndex = tokenInfoMsg.linkindex();

        reqCtx.result.readDiscoveryTokenResp.tokenInfo.push_back( dcgmTokenInfo );
    }
}

void
DcgmFMLWLinkIntf::handleNodeDisconnect(uint32 nodeId)
{
    // Note: Add another map (map <uint32, std::list<DcgmLWLinkReqCtx> > to track per node id
    // if the below walking is expensive (many outstanding trin req for a node)

    // node socket connection got disconnected. complete all the
    // outstanding training request to that master node as completed.
    // erase all the book keeping informations.
    DcgmFMAutoLock lock(mLock);

    PRINT_DEBUG( "", "DcgmFMLWLinkIntf handleNodeDisconnect\n" );

    TrainRequestMap::iterator it = mTrainReqPending.begin();

    while ( it != mTrainReqPending.end() ) {
        DcgmLWLinkReqCtx reqCtx = it->second;
        if ( reqCtx.toNodeId == nodeId ) {
            // move to the completed list as failure
            uint64 reqId = it->first;
            reqCtx.result.requestId = reqId;
            reqCtx.result.status = FM_LWLINK_ST_MASTER_FM_SOCKET_ERR;
            mTrainReqComplete.insert( std::make_pair(reqId, reqCtx) );
            mTrainReqPending.erase( it++ );
            continue;
        }
        ++it;
    }
}

void
DcgmFMLWLinkIntf::dumpTrainCtxEntry(std::ostream *os,
                                    uint64 reqId,
                                    DcgmLWLinkReqCtx &reqCtx)
{
    *os << "\t\trequestId:  " << reqId << std::endl;

    *os << "\t\tStatus:  " << reqCtx.result.status<< std::endl;

    switch(reqCtx.reqType) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            DcgmLWLinkConnTrainResp connTrainResp = reqCtx.result.connTrainResp;
            *os << "\t\tMasterlinkMode:  " << connTrainResp.masterState.linkMode << std::endl;
            *os << "\t\tMasterTxSubLinkMode:  " << connTrainResp.masterState.txSubLinkMode << std::endl;
            *os << "\t\tMasterRxSubLinkMode:  " << connTrainResp.masterState.rxSubLinkMode << std::endl;
            *os << "\t\tSlavelinkMode:  " << connTrainResp.slaveState.linkMode << std::endl;
            *os << "\t\tSlaveTxSubLinkMode:  " << connTrainResp.slaveState.txSubLinkMode << std::endl;
            *os << "\t\tSlaveRxSubLinkMode:  " << connTrainResp.slaveState.rxSubLinkMode << std::endl;
            break;
        }
        default: {
            break;
        }
    }    
}
