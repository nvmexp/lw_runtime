/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <stdlib.h>
#include "fm_log.h"
#include "GlobalFMLWLinkIntf.h"
#include "FMAutoLock.h"
#include <g_lwconfig.h>
#include <climits>

GlobalFMLWLinkIntf::GlobalFMLWLinkIntf(FMConnInterface *ctrlConnIntf)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf::GlobalFMLWLinkIntf" );

    lwosInitializeCriticalSection( &mLock );
    mCtrlConnIntf = ctrlConnIntf;
    mNextReqId = 0; // valid IDs start from 1
    mTrainReqPending.clear();
    mTrainReqComplete.clear();
}

GlobalFMLWLinkIntf::~GlobalFMLWLinkIntf()
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf::GlobalFMLWLinkIntf" );
    // since the map has stack objects, just empty the map container
    mTrainReqPending.clear();
    mTrainReqComplete.clear();
    lwosDeleteCriticalSection( &mLock );
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendResetSwitchLinksReq(FMLWLinkReq &linkReq,
                                            uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_RESET_SWITCH_LINKS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendResetAllSwitchLinksReq(FMLWLinkReq &linkReq,
                                               uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendConnTrainReq(FMLWLinkReq &linkReq,
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
        case LWLINK_TRAIN_SAFE_TO_HIGH_SUBLINK: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_HIGH_MAINLINK: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK, linkReq, requestId);
        }
        //TODO: add requests for sublink and mainlink training for other states linke HIGH_TO_SAFE, SAFE_TO_OFF etc
        default: {
            break;
        }
    } 

    return FM_INT_ST_BADPARAM;
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendConnTrainParallelReq(FMLWLinkReq &linkReq,
                                             uint64 &requestId)
{
    switch( linkReq.connTrainParallelReq[0].trainTo ) {
        case LWLINK_TRAIN_OFF_TO_SAFE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_HIGH: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH, linkReq, requestId);
        }
        case LWLINK_TRAIN_TO_OFF: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF, linkReq, requestId);
        }
        case LWLINK_TRAIN_HIGH_TO_SAFE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_OFF: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF, linkReq, requestId);
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case LWLINK_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_TO_OFF: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF, linkReq, requestId);
        }
        case LWLINK_TRAIN_SAFE_TO_INITOPTIMIZE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE, linkReq, requestId);
        }
        case LWLINK_TRAIN_POST_INITOPTIMIZE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ, linkReq, requestId);
        }
        case LWLINK_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS, linkReq, requestId);
        }	
        case LWLINK_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES: {
            return sendTrainRequest(lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES, linkReq, requestId);
        }	
        case LWLINK_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE: {
            return sendTrainRequest(lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE, linkReq, requestId);
        }
#endif
        default: {
            break;
        }
    } 

    return FM_INT_ST_BADPARAM;
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendGetDeviceLwlinkStateReq(FMLWLinkReq &linkReq, uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendDiscoverIntraNodeConnReq(FMLWLinkReq &linkReq,
                                                 uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS, linkReq, requestId);
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
FMIntReturn_t
GlobalFMLWLinkIntf::sendAddInterNodeConnReq(FMLWLinkReq &linkReq,
                                            uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_ADD_INTERNODE_CONN, linkReq, requestId);
}
#endif

FMIntReturn_t
GlobalFMLWLinkIntf::sendGetIntraNodeConnReq(FMLWLinkReq &linkReq,
                                            uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_GET_INTRANODE_CONNS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendWriteDiscoveryReq(FMLWLinkReq &linkReq,
                                          uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendReadSidReq(FMLWLinkReq &linkReq,
                                   uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_READ_SIDS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendReadDiscoveryReq(FMLWLinkReq &linkReq,
                                         uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS, linkReq, requestId);
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendSwitchTrainingFailedLinkInfo(FMLWLinkReq &linkReq,
                                                     uint64 &requestId)
{
    return sendTrainRequest(lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO, linkReq, requestId);
}

bool
GlobalFMLWLinkIntf::isLinkReqComplete(uint64 requestId,
                                      FMLWLinkReqResult &reqResult)
{
    FMAutoLock lock(mLock);
    TrainRequestMap::iterator it;

    it = mTrainReqComplete.find( requestId );
    if ( it != mTrainReqComplete.end() ) {
        FMLWLinkReqCtx reqCtx = it->second;
        reqResult = reqCtx.result;
        mTrainReqComplete.erase( requestId );
        return true;
    }
    return false;
}

void
GlobalFMLWLinkIntf::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf handleMessage" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE: {
            handleTrainReqResponseMsg( pFmMessage );
            break;
        }
        default: {
            FM_LOG_WARNING( "LWLink training interface received unknown message type" );
            break;
        }
    }
}

void
GlobalFMLWLinkIntf::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf handleEvent" );

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
GlobalFMLWLinkIntf::dumpInfo(std::ostream *os)
{
    FMAutoLock lock(mLock);

    *os << "\t\tGlobalFMLWLinkIntf Dump Start" << std::endl;

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
    *os << "\t\tGlobalFMLWLinkIntf Dump End" << std::endl;
}

FMIntReturn_t
GlobalFMLWLinkIntf::sendTrainRequest(lwswitch::FabricManagerMessageType msgType,
                                   FMLWLinkReq &linkReq,
                                   uint64 &requestId)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf sendTrainRequest msgType=%d", msgType );
    uint64 reqId = 0;
    FMIntReturn_t retVal;
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
		case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK : {
            toNodeId = ((linkReq.connTrainReq.masterNodeId != INT_MAX) ? 
                         linkReq.connTrainReq.masterNodeId : linkReq.connTrainReq.slaveNodeId);
            genConnTrainReqMsg( reqMsg, linkReq );
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
            // TODO remove once internode parallel link training has been stable for some time
            FM_LOG_DEBUG("FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF linkReq.connTrainParallelReq.size=%lu",
                          linkReq.connTrainParallelReq.size());
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
#endif
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS:
        case lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
#endif
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF: {
            toNodeId = linkReq.connTrainParallelReq[0].masterNodeId;            
            genConnTrainParallelReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_INITPHASE5:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS:
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN:
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST:
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
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            toNodeId = linkReq.addInterNodeConnReq.localEndInfo.nodeId; 
            genAddInterNodeConnReqMsg( reqMsg, linkReq );
            break;
        }
#endif
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
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO:{
            toNodeId = linkReq.switchTrainingFailedLinkInfoReq.nodeId;
            genSwitchTrainingFailedReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_READ_SIDS:{
            toNodeId = linkReq.readSidReq.nodeId; 
            genReadSidReqMsg( reqMsg, linkReq );
            break;
        }
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE: {
            toNodeId = linkReq.getDeviceLwlinkStateReq.nodeId; 
            genGetDeviceLwlinkStateReqMsg( reqMsg, linkReq );
            break;
        }
        default:{
            FM_LOG_ERROR( "LWLink interface is called to send unsupported LWLink request type %d", msgType);
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
    if ( retVal != FM_INT_ST_OK ) {
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
GlobalFMLWLinkIntf::genConnTrainReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                       FMLWLinkReq &linkReq)
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
GlobalFMLWLinkIntf::genConnTrainParallelReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                               FMLWLinkReq &linkReq)
{
    FM_LOG_DEBUG("GlobalFMLWLinkIntf::genConnTrainParallelReqMsg");
    lwswitch::lwlinkTrainParallelConnReqMsg *trainParallelConnReqMsg = new lwswitch::lwlinkTrainParallelConnReqMsg();
    for(auto it = linkReq.connTrainParallelReq.begin(); it != linkReq.connTrainParallelReq.end(); it++) {
        lwswitch::lwlinkEndPointInfo *masterEnd = new lwswitch::lwlinkEndPointInfo();
        lwswitch::lwlinkEndPointInfo *slaveEnd = new lwswitch::lwlinkEndPointInfo();
        masterEnd->set_nodeid( it->masterNodeId );
        masterEnd->set_gpuorswitchid( it->masterGpuOrSwitchId );
        masterEnd->set_linkindex( it->masterLinkIndex );
        slaveEnd->set_nodeid( it->slaveNodeId );
        slaveEnd->set_gpuorswitchid( it->slaveGpuOrSwitchId );
        slaveEnd->set_linkindex( it->slaveLinkIndex );

        // create the link connection pair
        lwswitch::lwlinkConnectionInfo *connInfo = trainParallelConnReqMsg->add_conninfo();
        connInfo->set_allocated_masterend( masterEnd );
        connInfo->set_allocated_slaveend( slaveEnd );

   }
   reqMsg->set_allocated_conntrainparallelreqmsg( trainParallelConnReqMsg );
}

void
GlobalFMLWLinkIntf::genNodeInitReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                      FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitReqMsg *nodeInitReqMsg = new lwswitch::lwlinkNodeInitReqMsg();
    reqMsg->set_allocated_nodeinitreqmsg( nodeInitReqMsg );
}

void
GlobalFMLWLinkIntf::genNodeInitStatusReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                            FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitStatusReqMsg *statusReqMsg = new lwswitch::lwlinkNodeInitStatusReqMsg();
    reqMsg->set_allocated_nodeinitstatusreqmsg( statusReqMsg );
}

void
GlobalFMLWLinkIntf::genNodeInitResetSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                      FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg *resetReqMsg = new lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg();
    resetReqMsg->set_switchphysicalid( linkReq.nodeInitResetSwitchLinksReq.switchId );
    resetReqMsg->set_linkmask( linkReq.nodeInitResetSwitchLinksReq.linkMask );
    reqMsg->set_allocated_nodeinitresetswitchlinksreqmsg( resetReqMsg );
}

void
GlobalFMLWLinkIntf::genNodeInitResetAllSwitchLinksReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                         FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkNodeInitResetAllSwitchLinksReqMsg *resetSwitchesReqMsg = new lwswitch::lwlinkNodeInitResetAllSwitchLinksReqMsg();
    reqMsg->set_allocated_nodeinitresetallswitchlinksreqmsg( resetSwitchesReqMsg );
}

void
GlobalFMLWLinkIntf::genDiscoverIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                   FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkDiscoverIntraNodeConnReqMsg *discoverConnReqMsg = new lwswitch::lwlinkDiscoverIntraNodeConnReqMsg();
    reqMsg->set_allocated_discoverintranodeconnreqmsg( discoverConnReqMsg );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

void
GlobalFMLWLinkIntf::genAddInterNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                              FMLWLinkReq &linkReq)
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
#endif

void
GlobalFMLWLinkIntf::genGetIntraNodeConnReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                              FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkGetIntraNodeConnReqMsg *getConnReqMsg = new lwswitch::lwlinkGetIntraNodeConnReqMsg();
    reqMsg->set_allocated_getintranodeconnreqmsg( getConnReqMsg );
}

void
GlobalFMLWLinkIntf::genWriteDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                            FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkWriteDiscoveryTokenReqMsg *writeDiscReqMsg = new lwswitch::lwlinkWriteDiscoveryTokenReqMsg();
    reqMsg->set_allocated_writedisctokenreqmsg( writeDiscReqMsg );
}

void
GlobalFMLWLinkIntf::genReadDiscoveryReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                           FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkReadDiscoveryTokenReqMsg *readDiscReqMsg = new lwswitch::lwlinkReadDiscoveryTokenReqMsg();
    reqMsg->set_allocated_readdisctokenreqmsg( readDiscReqMsg );
}

void
GlobalFMLWLinkIntf::genSwitchTrainingFailedReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                  FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkSwitchTrainingFailedReqMsg *trainingFailedReqMsg = new lwswitch::lwlinkSwitchTrainingFailedReqMsg();
    trainingFailedReqMsg->set_switchphysicalid( linkReq.switchTrainingFailedLinkInfoReq.switchId );
    trainingFailedReqMsg->set_trainingattemptedmask0( linkReq.switchTrainingFailedLinkInfoReq.attemptedMask0 );
    trainingFailedReqMsg->set_trainingfailedmask0( linkReq.switchTrainingFailedLinkInfoReq.failedMask0 );
    reqMsg->set_allocated_switchtrainingfailedreqmsg( trainingFailedReqMsg );
}

void
GlobalFMLWLinkIntf::genReadSidReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                     FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkReadSidReqMsg *readSidReqMsg = new lwswitch::lwlinkReadSidReqMsg();
    reqMsg->set_allocated_readsidreqmsg( readSidReqMsg );
}

void
GlobalFMLWLinkIntf::genGetDeviceLwlinkStateReqMsg(lwswitch::lwlinkRequestMsg *reqMsg,
                                                  FMLWLinkReq &linkReq)
{
    lwswitch::lwlinkGetDeviceLwlinkStateReqMsg *getDeviceLwlinkStateReqMsg = new lwswitch::lwlinkGetDeviceLwlinkStateReqMsg();
    reqMsg->set_allocated_getdevicelwlinkstatereqmsg( getDeviceLwlinkStateReqMsg );
}

uint64
GlobalFMLWLinkIntf::getNextTrainReqId(void)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf getNextTrainReqId" );
    FMAutoLock lock(mLock);

    mNextReqId++;
    if ( mNextReqId == 0 ) {
        // wrap around
        mNextReqId = 1;
    }
    return mNextReqId;
}

void
GlobalFMLWLinkIntf::addToReqPendingTable(uint64 reqId,
                                         FMLWLinkReq &linkReq,
                                         lwswitch::FabricManagerMessageType reqType,
                                         uint32 toNodeId)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf addToReqPendingTable" );
    FMAutoLock lock(mLock);

    FMLWLinkReqCtx reqCtx = {0};
    reqCtx.reqType = reqType;
    reqCtx.toNodeId = toNodeId;
    reqCtx.req = linkReq;

    if ( mTrainReqPending.count(reqId) ) {
        FM_LOG_WARNING( "LWLink train request with request id %llu already present in the tracking table", reqId );
        return;
    }

    mTrainReqPending.insert( std::make_pair(reqId, reqCtx) );
}

void
GlobalFMLWLinkIntf::removeFromReqPendingTable(uint64 reqId)
{
    FMAutoLock lock(mLock);

    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it != mTrainReqPending.end() ) {
        mTrainReqPending.erase( it );
    }
}

bool
GlobalFMLWLinkIntf::getPendingTrainReq(uint64 reqId, FMLWLinkReqCtx &reqCtx)
{
    FMAutoLock lock(mLock);

    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it == mTrainReqPending.end() ) {
        return false;
    }

    // found the request, copy it and return success
    reqCtx = it->second;
    return true;
}

void
GlobalFMLWLinkIntf::markPendingReqAsComplete(uint64 reqId, FMLWLinkReqCtx &reqCtx)
{
    FMAutoLock lock(mLock);
    TrainRequestMap::iterator it = mTrainReqPending.find( reqId );
    if ( it == mTrainReqPending.end() ) {
        return;
    }

    // erase from pending and add to complete list
    mTrainReqPending.erase( it );
    mTrainReqComplete.insert( std::make_pair(reqId, reqCtx) );
}

void
GlobalFMLWLinkIntf::handleTrainReqResponseMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "GlobalFMLWLinkIntf handleTrainReqResponseMsg" );

    // find the corresponding link training request_id and save the result
    // also move the request to the completed requests table

    if ( !pFmMessage->has_lwlinkmsg() ) {
        FM_LOG_WARNING( "received LWLink training complete message without required fields" );
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    if ( !linkMsg.has_rspmsg() ) {
        FM_LOG_WARNING( "received LWLink training complete message without the response message" );
        return;
    }

    uint64 reqId = linkMsg.trainreqid();
    FMLWLinkReqCtx reqCtx = {0};
    if ( !getPendingTrainReq(reqId, reqCtx) ) {
        FM_LOG_WARNING( "no LWLink training request with request id %llu found during training response message handling", reqId );
        return;
    }

    // update our local context with result information
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();

    reqCtx.result.requestId = reqId;
    reqCtx.result.status = (LWLinkErrorCodes)rspMsg.status();

    switch(reqCtx.reqType) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK: {
            parseConnTrainResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
        case lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
#endif
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF: {
            parseConnTrainParallelResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_INITPHASE5:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS:
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN:
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST:
#endif
        case lwswitch::FM_LWLINK_INIT:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:    
#endif
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO: {
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
        case lwswitch::FM_LWLINK_READ_SIDS:{
            parseReadSidResp(rspMsg, reqCtx);
            break;
        }
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE: {
            parseGetDeviceLwlinkStateResp(rspMsg, reqCtx);
            break;
        }
        default:{
            FM_LOG_ERROR( "unknown LWLink request type %d while parsing training response message", reqCtx.reqType);
            break;
        }
    }

    // remove from outstanding request and add to completed req table
    markPendingReqAsComplete(reqId, reqCtx);
}

void
GlobalFMLWLinkIntf::parseConnTrainParallelResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                               FMLWLinkReqCtx &reqCtx)
{
    lwswitch::lwlinkTrainParallelConnRspMsg connRspMsg = rspMsg.conntrainparallelrspmsg();
    for ( int i = 0; i < connRspMsg.connrspinfo_size(); i++ ) {
        auto conninfo = connRspMsg.connrspinfo(i);
        FMLWLinkConnTrainRespDetailed connTrainParallelResp;

        connTrainParallelResp.masterNodeId = conninfo.masterend().nodeid();
        connTrainParallelResp.masterGpuOrSwitchId = conninfo.masterend().gpuorswitchid();
        connTrainParallelResp.masterLinkIndex = conninfo.masterend().linkindex();
        connTrainParallelResp.slaveNodeId = conninfo.slaveend().nodeid();
        connTrainParallelResp.slaveGpuOrSwitchId = conninfo.slaveend().gpuorswitchid();
        connTrainParallelResp.slaveLinkIndex = conninfo.slaveend().linkindex();


        connTrainParallelResp.masterState.linkMode = conninfo.masterend().state().linkmode();
        connTrainParallelResp.masterState.txSubLinkMode = conninfo.masterend().state().txsublinkmode();
        connTrainParallelResp.masterState.rxSubLinkMode = conninfo.masterend().state().rxsublinkmode();
        connTrainParallelResp.slaveState.linkMode = conninfo.slaveend().state().linkmode();
        connTrainParallelResp.slaveState.txSubLinkMode = conninfo.slaveend().state().txsublinkmode();
        connTrainParallelResp.slaveState.rxSubLinkMode = conninfo.slaveend().state().rxsublinkmode();

        connTrainParallelResp.masterQualityInfo.eomLow = conninfo.masterend().state().qualityinfo().eomlow();
        connTrainParallelResp.slaveQualityInfo.eomLow = conninfo.slaveend().state().qualityinfo().eomlow();

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        // parse FOM
        connTrainParallelResp.fomValues.numLanes = conninfo.masterend().state().fomvalues().numlanes();
        for(int j = 0; j < conninfo.masterend().state().fomvalues().fomvalues_size(); j++)
            connTrainParallelResp.fomValues.fomValues[j] = conninfo.masterend().state().fomvalues().fomvalues(j);

        // parse grading values
        connTrainParallelResp.gradingValues.laneMask = conninfo.masterend().state().gradingvalues().lanemask();
        for(int j = 0; j < conninfo.masterend().state().gradingvalues().txinit_size(); j++) {
            connTrainParallelResp.gradingValues.txInit[j] = conninfo.masterend().state().gradingvalues().txinit(j);
            connTrainParallelResp.gradingValues.rxInit[j] = conninfo.masterend().state().gradingvalues().rxinit(j);
            connTrainParallelResp.gradingValues.txMaint[j] = conninfo.masterend().state().gradingvalues().txmaint(j);
            connTrainParallelResp.gradingValues.rxMaint[j] = conninfo.masterend().state().gradingvalues().rxmaint(j);
        }
#endif
        reqCtx.result.connTrainParallelResp.push_back(connTrainParallelResp);
    }
}

void
GlobalFMLWLinkIntf::parseConnTrainResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                       FMLWLinkReqCtx &reqCtx)
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
GlobalFMLWLinkIntf::parseLinkInitStatusResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                            FMLWLinkReqCtx &reqCtx)
{
    lwswitch::lwlinkNodeInitStatusRspMsg statusRspMsg = rspMsg.nodeinitstatusrspmsg();

    // parse status for each device and its links.
    for ( int idx = 0; idx < statusRspMsg.initstatus_size(); idx++ ) {
        lwswitch::lwlinkDeviceLinkInitStatus devStatus = statusRspMsg.initstatus( idx );
        FMLinkInitStatusInfo linkStatusInfo;
        linkStatusInfo.nodeId = statusRspMsg.nodeid();
        linkStatusInfo.gpuOrSwitchId = devStatus.gpuorswitchid();
        for ( int linkIdx = 0; linkIdx < devStatus.linkstatus_size(); linkIdx++ ) {
            lwswitch::lwlinkLinkInitStatus linkStatus = devStatus.linkstatus( linkIdx );
            linkStatusInfo.initStatus[linkIdx].linkIndex = linkStatus.linkindex();
            linkStatusInfo.initStatus[linkIdx].initStatus = linkStatus.status();            
         }
        reqCtx.result.nodeInitStatusResp.statusInfo.push_back( linkStatusInfo );
    }
}

void
GlobalFMLWLinkIntf::parseGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                              FMLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkGetIntraNodeConnRspMsg getRspMsg = rspMsg.getintranodeconnrspmsg();

    for (idx = 0; idx < getRspMsg.conninfo_size(); idx++) {

        lwswitch::lwlinkConnectionInfo connInfoMsg = getRspMsg.conninfo(idx);
        lwswitch::lwlinkEndPointInfo masterEnd = connInfoMsg.masterend();
        lwswitch::lwlinkEndPointInfo slaveEnd = connInfoMsg.slaveend();

        // parase the google protobuf msg and move to plain format
        FMLWLinkConnInfo linkConnInfo;
        linkConnInfo.masterEnd.nodeId = masterEnd.nodeid();
        linkConnInfo.masterEnd.linkIndex = masterEnd.linkindex();
        linkConnInfo.masterEnd.gpuOrSwitchId = masterEnd.gpuorswitchid();

        linkConnInfo.slaveEnd.nodeId = slaveEnd.nodeid();
        linkConnInfo.slaveEnd.linkIndex = slaveEnd.linkindex();
        linkConnInfo.slaveEnd.gpuOrSwitchId = slaveEnd.gpuorswitchid();

        reqCtx.result.getIntraNodeConnResp.connInfo.push_back( linkConnInfo );
    }
}

void
GlobalFMLWLinkIntf::parseWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                                 FMLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkWriteDiscoveryTokenRspMsg discRspMsg = rspMsg.writedisctokenrspmsg();

    for (idx = 0; idx < discRspMsg.tokeninfo_size(); idx++) {

        lwswitch::lwlinkDiscoveryTokenInfo tokenInfoMsg = discRspMsg.tokeninfo(idx);

        // parase the google protobuf msg and move to plain format
        FMLinkDiscoveryTokenInfo linkTokenInfo;
        linkTokenInfo.tokelwalue = tokenInfoMsg.tokelwalue();
        linkTokenInfo.nodeId = tokenInfoMsg.nodeid();
        linkTokenInfo.gpuOrSwitchId = tokenInfoMsg.gpuorswitchid();
        linkTokenInfo.linkIndex = tokenInfoMsg.linkindex();

        reqCtx.result.writeDiscoveryTokenResp.tokenInfo.push_back( linkTokenInfo );
    }
}

void
GlobalFMLWLinkIntf::parseReadSidResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                     FMLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkReadSidRspMsg sidRspMsg = rspMsg.readsidrspmsg();
    for (idx = 0; idx < sidRspMsg.sidinfo_size(); idx++) {

        lwswitch::lwlinkSidInfo sidInfoMsg = sidRspMsg.sidinfo(idx);

        // parse the google protobuf msg and move to plain format
        FMLinkSidInfo sidInfo;
        sidInfo.nodeId = sidInfoMsg.nodeid();
        sidInfo.gpuOrSwitchId = sidInfoMsg.gpuorswitchid();

        sidInfo.nearSid = sidInfoMsg.nearsid();
        sidInfo.nearLinkIndex = sidInfoMsg.nearlinkindex();
        sidInfo.farSid = sidInfoMsg.farsid();
        sidInfo.farLinkIndex = sidInfoMsg.farlinkindex();

        reqCtx.result.readSidResp.sidList.push_back( sidInfo );
    }
}


void
GlobalFMLWLinkIntf::parseReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                                FMLWLinkReqCtx &reqCtx)
{
    int idx = 0;
    lwswitch::lwlinkReadDiscoveryTokenRspMsg discRspMsg = rspMsg.readdisctokenrspmsg();

    for (idx = 0; idx < discRspMsg.tokeninfo_size(); idx++) {

        lwswitch::lwlinkDiscoveryTokenInfo tokenInfoMsg = discRspMsg.tokeninfo(idx);

        // parase the google protobuf msg and move to plain format
        FMLinkDiscoveryTokenInfo linkTokenInfo;
        linkTokenInfo.tokelwalue = tokenInfoMsg.tokelwalue();
        linkTokenInfo.nodeId = tokenInfoMsg.nodeid();
        linkTokenInfo.gpuOrSwitchId = tokenInfoMsg.gpuorswitchid();
        linkTokenInfo.linkIndex = tokenInfoMsg.linkindex();

        reqCtx.result.readDiscoveryTokenResp.tokenInfo.push_back( linkTokenInfo );
    }
}

void
GlobalFMLWLinkIntf::parseGetDeviceLwlinkStateResp(lwswitch::lwlinkResponseMsg &rspMsg, 
                                                  FMLWLinkReqCtx &reqCtx)
{
    int idx;
    lwswitch::lwlinkGetDeviceLwlinkStateRspMsg getRspMsg = rspMsg.getdevicelwlinkstaterspmsg();

    for (idx = 0; idx < getRspMsg.lwendinfo_size(); idx++) {
        lwswitch::lwlinkEndPointInfo lwEndInfoMsg = getRspMsg.lwendinfo(idx);

        // parse the google protobuf msg and move to plain format
        FMLWLinkGetDeviceLwlinkStateRespDetailed getDeviceLwlinkStateInfo;

        getDeviceLwlinkStateInfo.lwEndInfo.nodeId = lwEndInfoMsg.nodeid();
        getDeviceLwlinkStateInfo.lwEndInfo.gpuOrSwitchId = lwEndInfoMsg.gpuorswitchid();
        getDeviceLwlinkStateInfo.lwEndInfo.linkIndex = lwEndInfoMsg.linkindex();
        getDeviceLwlinkStateInfo.stateInfo.linkMode = lwEndInfoMsg.state().linkmode();
        getDeviceLwlinkStateInfo.stateInfo.txSubLinkMode = lwEndInfoMsg.state().txsublinkmode();
        getDeviceLwlinkStateInfo.stateInfo.rxSubLinkMode = lwEndInfoMsg.state().rxsublinkmode();

        reqCtx.result.getDeviceLwlinkStateResp.push_back( getDeviceLwlinkStateInfo );
    }
}

void
GlobalFMLWLinkIntf::handleNodeDisconnect(uint32 nodeId)
{
    // Note: Add another map (map <uint32, std::list<FMLWLinkReqCtx> > to track per node id
    // if the below walking is expensive (many outstanding trin req for a node)

    // node socket connection got disconnected. complete all the
    // outstanding training request to that master node as completed.
    // erase all the book keeping informations.
    FMAutoLock lock(mLock);

    FM_LOG_DEBUG( "GlobalFMLWLinkIntf handleNodeDisconnect" );

    TrainRequestMap::iterator it = mTrainReqPending.begin();

    while ( it != mTrainReqPending.end() ) {
        FMLWLinkReqCtx reqCtx = it->second;
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
GlobalFMLWLinkIntf::dumpTrainCtxEntry(std::ostream *os,
                                      uint64 reqId,
                                      FMLWLinkReqCtx &reqCtx)
{
    *os << "\t\trequestId:  " << reqId << std::endl;

    *os << "\t\tStatus:  " << reqCtx.result.status<< std::endl;

    switch(reqCtx.reqType) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            FMLWLinkConnTrainResp connTrainResp = reqCtx.result.connTrainResp;
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
