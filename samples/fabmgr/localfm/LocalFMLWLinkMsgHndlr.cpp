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

#include "fm_log.h"
#include <sstream>

#include "LocalFMLWLinkMsgHndlr.h"

#include "LocalFMLWLinkReqInit.h"
#include "LocalFMLWLinkReqConn.h"
#include "LocalFMLWLinkReqDiscovery.h"
#include "FMLWLinkTypes.h"
#include "LocalFMLWLinkReqConnTrainParallel.h"
#include <g_lwconfig.h>


LocalFMLWLinkMsgHndlr::LocalFMLWLinkMsgHndlr(FMConnInterface *ctrlConnIntf,
                                             LocalFMLWLinkDrvIntf *linkDrvIntf,
                                             LocalFMLWLinkDevRepo *linkDevRepo)
{
    mCtrlConnIntf = ctrlConnIntf;
    mLWLinkDrvIntf = linkDrvIntf;
    mLWLinkDevRepo = linkDevRepo;
    mFMTrainReqMap.clear();
}

LocalFMLWLinkMsgHndlr::~LocalFMLWLinkMsgHndlr()
{
    // erase all the pending requests.
    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    while ( it != mFMTrainReqMap.end() ) {
        LocalFMLWLinkReqBase* fmTrainReq = it->second;
        mFMTrainReqMap.erase( it++ );        
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleMessage(lwswitch::fmMessage* pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: 
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
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
            handleMasterConnTrainReqMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            handleSlaveConnTrainReqMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM: {
            handleSlaveConfirmMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE: {
            handleSlaveCompletemMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_TRAIN_RSP_MASTER_SYNC: {
            handleMasterSyncMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_SYNC: {
            handleSlaveSyncMsg( pFmMessage );
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
        case lwswitch::FM_LWLINK_INIT_STATUS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO:
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE: {
            handleInitMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS: 
        case lwswitch::FM_LWLINK_READ_SIDS:
        {
            handleDiscoverMsg( pFmMessage );
            break;
        }
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {            
            handleConnectionMsg( pFmMessage );
            break;
        }
        default: {
            FM_LOG_WARNING( "link training message handler received an unknown message type %d\n",
                             pFmMessage->type() );
            break;
        }
    }
}

void
LocalFMLWLinkMsgHndlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleEvent\n" );

    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // do nothing
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            handleNodeDisconnect( nodeId );
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // we lost connectivity with GFM. 
            // GFM will mark the outstanding request as failed.
            // we just clean-up all our master requests
            handleGFMDisconnect();
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // do nothing
            break;
        }
    }
}

void
LocalFMLWLinkMsgHndlr::handleMasterConnTrainReqMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleMasterConnTrainReqMsg\n");
    LocalFMLWLinkReqBase *fmTrainReq = NULL;
    bool bReqCompleted = false;
    uint32 trainType = pFmMessage->type();

    // allocate the LWLink request type based on the GPB message.        
    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            fmTrainReq = new LocalFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
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
            fmTrainReq = new LocalFMLWLinkReqConnTrainParallel( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        default: {

            return;
        }    
    }

    // process the new request event
    addToReqTrackingTable( fmTrainReq );

    deliverMessage( fmTrainReq, MASTER_REQ_RECVD, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleSlaveConnTrainReqMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleSlaveConnTrainReqMsg\n" );
    LocalFMLWLinkReqConnBase *fmTrainReq;
    bool bReqCompleted = false;
    uint32 trainType = pFmMessage->type();

    // allocate the LWLink request type based on the GPB message.        
    switch ( trainType ) {
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            fmTrainReq = new LocalFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        default: {
            return;
        }    
    }

    addToReqTrackingTable( fmTrainReq );

    deliverMessage( fmTrainReq, SLAVE_REQ_RECVD, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleInitMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleSlaveLinkConnReqMsg\n" );

    LocalFMLWLinkReqInit *fmTrainReq = new LocalFMLWLinkReqInit( pFmMessage, mCtrlConnIntf,
                                                               mLWLinkDrvIntf, mLWLinkDevRepo );

    addToReqTrackingTable( fmTrainReq );

    deliverMessage( fmTrainReq, MASTER_REQ_RECVD, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleDiscoverMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleDiscoverMsg\n" );

    LocalFMLWLinkReqDiscovery *fmTrainReq = new LocalFMLWLinkReqDiscovery( pFmMessage, mCtrlConnIntf,
                                                                         mLWLinkDrvIntf, mLWLinkDevRepo );

    addToReqTrackingTable( fmTrainReq );

    deliverMessage( fmTrainReq, MASTER_REQ_RECVD, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleConnectionMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleConnectionMsg\n" );

    LocalFMLWLinkReqConn *fmTrainReq = new LocalFMLWLinkReqConn( pFmMessage, mCtrlConnIntf,
                                                               mLWLinkDrvIntf, mLWLinkDevRepo );

    addToReqTrackingTable( fmTrainReq );

    deliverMessage( fmTrainReq, MASTER_REQ_RECVD, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleSlaveConfirmMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleSlaveConfirmMsg msgType = %d\n", pFmMessage->type() );
    bool bReqCompleted;
    LocalFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        FM_LOG_WARNING("received link training slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        FM_LOG_WARNING("received link training slave confirmation for request %llu. But no corresponding master request\n", reqId);
        return;
    }

    deliverMessage( fmTrainReq, SLAVE_RESP_CONFIRM, bReqCompleted, pFmMessage );
    if( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleSlaveCompletemMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleSlaveCompletemMsg\n" );
    bool bReqCompleted;
    LocalFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        FM_LOG_WARNING("received link training slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        FM_LOG_WARNING("received link training slave complete for request %llu. But no corresponding master request\n", reqId);
        return;
    }

    deliverMessage( fmTrainReq, SLAVE_RESP_COMPLETE, bReqCompleted, pFmMessage );
    FM_LOG_DEBUG( "reqId=%lld bReqCompleted=%d type=%d\n", reqId, bReqCompleted, pFmMessage->type() );
    if( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleSlaveSyncMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleSlaveSyncMsg\n" );
    bool bReqCompleted;
    LocalFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        FM_LOG_WARNING("received link training slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        FM_LOG_WARNING("received link training slave sync for request %llu. But no corresponding master request\n", reqId);
        return;
    }

    deliverMessage( fmTrainReq, SLAVE_RESP_SYNC, bReqCompleted, pFmMessage );
    if( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::handleMasterSyncMsg(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleMasterSyncMsg\n" );
    bool bReqCompleted;
    LocalFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        FM_LOG_WARNING("received link training slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        FM_LOG_WARNING("received link training master sync for request %llu. But no corresponding master request\n", reqId);
        return;
    }

    deliverMessage( fmTrainReq, MASTER_RESP_SYNC, bReqCompleted, pFmMessage );
    if ( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
LocalFMLWLinkMsgHndlr::deliverMessage(LocalFMLWLinkReqBase *fmTrainReq,
                                      TrainReqCallCtx callCtx,
                                      bool &bReqCompleted,
                                      lwswitch::fmMessage *pFmMessage)
{
    switch ( callCtx ) {
        case MASTER_REQ_RECVD: {
            bReqCompleted = fmTrainReq->processNewMasterRequest( pFmMessage );
            break;
        }
        case SLAVE_REQ_RECVD: {
            bReqCompleted = fmTrainReq->processNewSlaveRequest( pFmMessage );
            break;
        }
        case SLAVE_RESP_CONFIRM: {
            bReqCompleted = fmTrainReq->processRespConfirm( pFmMessage );
            break;
        }
        case SLAVE_RESP_COMPLETE: {
            bReqCompleted = fmTrainReq->processRespComplete( pFmMessage );
            break;
        }
        case SLAVE_RESP_SYNC: {
            bReqCompleted = fmTrainReq->processSlaveSync( pFmMessage );
            break;
        }
        case MASTER_RESP_SYNC: {
            bReqCompleted = fmTrainReq->processMasterSync( pFmMessage );
            break;
        }
        case REQ_TIMED_OUT: {
            bReqCompleted = fmTrainReq->processReqTimeOut();
            break;
        }
        default: {
            FM_LOG_WARNING( "invalid call context for link training request \n" );
        }
    }
}

void
LocalFMLWLinkMsgHndlr::addToReqTrackingTable(LocalFMLWLinkReqBase *fmTrainReq)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr addToReqTrackingTable\n" );
    uint64 reqId = fmTrainReq->getTrainReqId();

    if ( mFMTrainReqMap.count( reqId) ) {
        FM_LOG_WARNING( "link training request with train request ID %llu already present\n", reqId );
        return;
    }

    mFMTrainReqMap.insert( std::make_pair(reqId, fmTrainReq) );
}

void
LocalFMLWLinkMsgHndlr::removeFromReqTrackingTable(uint64 reqId)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr removeFromReqTrackingTable\n" );
    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        FM_LOG_WARNING( "no link training request with specified train request ID %llu \n", reqId );
        return;
    }

    mFMTrainReqMap.erase( it );
}

void
LocalFMLWLinkMsgHndlr::removeFromReqTrackingTable(LocalFMLWLinkReqBase *fmTrainReq)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr removeFromReqTrackingTable\n" );
    uint64 reqId = fmTrainReq->getTrainReqId();

    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        FM_LOG_WARNING( "no link training request with specified train request ID %llu \n", reqId );
        return;
    }

    mFMTrainReqMap.erase( it );
}

LocalFMLWLinkReqBase*
LocalFMLWLinkMsgHndlr::getReqFromTrackingTable(uint64 reqId)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr getReqFromTrackingTable\n" );
    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        return NULL;
    } else {
        return it->second;
    }
}

void
LocalFMLWLinkMsgHndlr::handleNodeDisconnect(uint32 nodeId)
{
    // node socket connection got disconnected. complete all the 
    // outstanding training requests (from master FM perspective)
    // to this slave node FM

    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleNodeDisconnect\n" );
    bool bReqCompleted;

    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    while ( it != mFMTrainReqMap.end() ) {
        LocalFMLWLinkReqBase* fmTrainReq = it->second;
        // only for master request. initiate cleaning from
        // one side only (slave FM will see a socket disconnection)
        // but master is pro-actively failing the request
        if ( fmTrainReq->isMasterReq()) {
            // Note : only connection train has master/slave sync. So type casting to that
            LocalFMLWLinkReqConnTrain *connTrainReq = (LocalFMLWLinkReqConnTrain*)fmTrainReq;
            if ( connTrainReq->getSlaveNodeId() == nodeId ) {
                // TODO: need a new callctx type....
                deliverMessage( fmTrainReq, REQ_TIMED_OUT, bReqCompleted, NULL );
                mFMTrainReqMap.erase( it++ );
                delete fmTrainReq;
                continue;
            }                
        }
        ++it;
    }
}

void
LocalFMLWLinkMsgHndlr::handleGFMDisconnect(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkMsgHndlr handleGFMDisconnect\n" );

    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    // we lost connection with GFM

    // NOTE : lwrrently we are clearing all the request. However, we can
    // keep the peer LFM slave request if required. The assumption is that
    // if the master FM is getting disconnected, all the peer LFM will see
    // a master FM disconnection and all the outstanding requests will be
    // purged from all the nodes anyway.
    while ( it != mFMTrainReqMap.end() ) {
        LocalFMLWLinkReqBase* fmTrainReq = it->second;
        if ( fmTrainReq->isMasterReq() ) {
            mFMTrainReqMap.erase( it++ );
            delete fmTrainReq;
            continue;
        }
        ++it;
    }
}

void
LocalFMLWLinkMsgHndlr::dumpInfo(std::ostream *os)
{
    LocalFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    if ( it != mFMTrainReqMap.end() ) {
        (it->second)->dumpInfo( os );
    }
}

