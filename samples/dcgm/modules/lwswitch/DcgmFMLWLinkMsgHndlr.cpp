
#include "logging.h"
#include <sstream>

#include "DcgmFMLWLinkMsgHndlr.h"

#include "DcgmFMLWLinkReqInit.h"
#include "DcgmFMLWLinkReqConn.h"
#include "DcgmFMLWLinkReqDiscovery.h"
#include "DcgmFMLWLinkTypes.h"
#include <g_lwconfig.h>


DcgmFMLWLinkMsgHndlr::DcgmFMLWLinkMsgHndlr(FMConnInterface *ctrlConnIntf,
                                           DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                           DcgmLFMLWLinkDevRepo *linkDevRepo)
{
    mCtrlConnIntf = ctrlConnIntf;
    mLWLinkDrvIntf = linkDrvIntf;
    mLWLinkDevRepo = linkDevRepo;
    mFMTrainReqMap.clear();
}

DcgmFMLWLinkMsgHndlr::~DcgmFMLWLinkMsgHndlr()
{
    // erase all the pending requests.
    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    while ( it != mFMTrainReqMap.end() ) {
        DcgmFMLWLinkReqBase* fmTrainReq = it->second;
        mFMTrainReqMap.erase( it++ );        
        delete fmTrainReq;
    }
}

void
DcgmFMLWLinkMsgHndlr::handleMessage(lwswitch::fmMessage* pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
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
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
#endif
        case lwswitch::FM_LWLINK_INIT:
        case lwswitch::FM_LWLINK_INIT_STATUS:
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS:
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS: {
            handleInitMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS:
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS:
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS: {
            handleDiscoverMsg( pFmMessage );
            break;
        }
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN:
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {            
            handleConnectionMsg( pFmMessage );
            break;
        }
        default: {
            PRINT_WARNING( "", "Link training Msg Handler received unknown message type\n" );
            break;
        }
    }
}

void
DcgmFMLWLinkMsgHndlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleEvent\n" );

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
DcgmFMLWLinkMsgHndlr::handleMasterConnTrainReqMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleMasterConnTrainReqMsg\n" );
    DcgmFMLWLinkReqConnBase *fmTrainReq = NULL;
    bool bReqCompleted = false;
    uint32 trainType = pFmMessage->type();

    // allocate the LWLink request type based on the GPB message.        
    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
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
DcgmFMLWLinkMsgHndlr::handleSlaveConnTrainReqMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleSlaveConnTrainReqMsg\n" );
    DcgmFMLWLinkReqConnBase *fmTrainReq;
    bool bReqCompleted = false;
    uint32 trainType = pFmMessage->type();

    // allocate the LWLink request type based on the GPB message.        
    switch ( trainType ) {
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                      mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
                                                       mLWLinkDrvIntf, mLWLinkDevRepo );
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            fmTrainReq = new DcgmFMLWLinkReqConnTrain( pFmMessage, mCtrlConnIntf,
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
DcgmFMLWLinkMsgHndlr::handleInitMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleSlaveLinkConnReqMsg\n" );

    DcgmFMLWLinkReqInit *fmTrainReq = new DcgmFMLWLinkReqInit( pFmMessage, mCtrlConnIntf,
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
DcgmFMLWLinkMsgHndlr::handleDiscoverMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleDiscoverMsg\n" );

    DcgmFMLWLinkReqDiscovery *fmTrainReq = new DcgmFMLWLinkReqDiscovery( pFmMessage, mCtrlConnIntf,
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
DcgmFMLWLinkMsgHndlr::handleConnectionMsg(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted;

    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleConnectionMsg\n" );

    DcgmFMLWLinkReqConn *fmTrainReq = new DcgmFMLWLinkReqConn( pFmMessage, mCtrlConnIntf,
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
DcgmFMLWLinkMsgHndlr::handleSlaveConfirmMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleSlaveConfirmMsg\n" );
    bool bReqCompleted;
    DcgmFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_WARNING("", "received link slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        PRINT_WARNING("%llu", "received slave confirmation for req %llu. But no corresponding master request\n", reqId);
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
DcgmFMLWLinkMsgHndlr::handleSlaveCompletemMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleSlaveCompletemMsg\n" );
    bool bReqCompleted;
    DcgmFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_WARNING("", "received link slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        PRINT_WARNING("%llu", "received slave complete for req %llu. But no corresponding master request\n", reqId);
        return;
    }

    deliverMessage( fmTrainReq, SLAVE_RESP_COMPLETE, bReqCompleted, pFmMessage );
    PRINT_DEBUG( "%lld %d %d", "reqId=%lld bReqCompleted=%d type=%d\n", reqId, bReqCompleted, pFmMessage->type() );
    if( bReqCompleted ) {
        // request is completed, remove from tracking list
        // and free the associated request
        removeFromReqTrackingTable( fmTrainReq );
        delete fmTrainReq;
    }
}

void
DcgmFMLWLinkMsgHndlr::handleSlaveSyncMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleSlaveSyncMsg\n" );
    bool bReqCompleted;
    DcgmFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_WARNING("", "received link slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        PRINT_WARNING("%llu", "received slave sync for req %llu. But no corresponding master request\n", reqId);
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
DcgmFMLWLinkMsgHndlr::handleMasterSyncMsg(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleMasterSyncMsg\n" );
    bool bReqCompleted;
    DcgmFMLWLinkReqBase *fmTrainReq = NULL;

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_WARNING("", "received link slave confirm message without required fields\n");
        return;
    }

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    uint64 reqId = linkMsg.trainreqid();
    fmTrainReq = getReqFromTrackingTable( reqId );
    if ( fmTrainReq == NULL) {
        PRINT_WARNING("%llu", "received master sync for req %llu. But no corresponding master request\n", reqId);
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
DcgmFMLWLinkMsgHndlr::deliverMessage(DcgmFMLWLinkReqBase *fmTrainReq,
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
            PRINT_WARNING("", "Invalid call context for training request \n" );
        }
    }
}

void
DcgmFMLWLinkMsgHndlr::addToReqTrackingTable(DcgmFMLWLinkReqBase *fmTrainReq)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr addToReqTrackingTable\n" );
    uint64 reqId = fmTrainReq->getTrainReqId();

    if ( mFMTrainReqMap.count( reqId) ) {
        PRINT_WARNING( "%llu", "Train request with Train Request ID %llu already present\n", reqId );
        return;
    }

    mFMTrainReqMap.insert( std::make_pair(reqId, fmTrainReq) );
}

void
DcgmFMLWLinkMsgHndlr::removeFromReqTrackingTable(uint64 reqId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr removeFromReqTrackingTable\n" );
    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        PRINT_WARNING( "%llu", "No link train request with Train Request ID %llu \n", reqId );
        return;
    }

    mFMTrainReqMap.erase( it );
}

void
DcgmFMLWLinkMsgHndlr::removeFromReqTrackingTable(DcgmFMLWLinkReqBase *fmTrainReq)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr removeFromReqTrackingTable\n" );
    uint64 reqId = fmTrainReq->getTrainReqId();

    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        PRINT_WARNING( "%llu", "No link train request with Train Request ID %llu \n", reqId );
        return;
    }

    mFMTrainReqMap.erase( it );
}

DcgmFMLWLinkReqBase*
DcgmFMLWLinkMsgHndlr::getReqFromTrackingTable(uint64 reqId)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr getReqFromTrackingTable\n" );
    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.find( reqId );

    if ( it == mFMTrainReqMap.end() ) {
        return NULL;
    } else {
        return it->second;
    }
}

void
DcgmFMLWLinkMsgHndlr::handleNodeDisconnect(uint32 nodeId)
{
    // node socket connection got disconnected. complete all the 
    // outstanding training requests (from master FM perspective)
    // to this slave node FM

    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleNodeDisconnect\n" );
    bool bReqCompleted;

    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    while ( it != mFMTrainReqMap.end() ) {
        DcgmFMLWLinkReqBase* fmTrainReq = it->second;
        // only for master request. initiate cleaning from
        // one side only (slave FM will see a socket disconnection)
        // but master is pro-actively failing the request
        if ( fmTrainReq->isMasterReq()) {
            // Note : only connection train has master/slave sync. So type casting to that
            DcgmFMLWLinkReqConnTrain *connTrainReq = (DcgmFMLWLinkReqConnTrain*)fmTrainReq;
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
DcgmFMLWLinkMsgHndlr::handleGFMDisconnect(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkMsgHndlr handleGFMDisconnect\n" );

    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    // we lost connection with GFM

    // NOTE : lwrrently we are clearing all the request. However, we can
    // keep the peer LFM slave request if required. The assumption is that
    // if the master FM is getting disconnected, all the peer LFM will see
    // a master FM disconnection and all the outstanding requests will be
    // purged from all the nodes anyway.
    while ( it != mFMTrainReqMap.end() ) {
        DcgmFMLWLinkReqBase* fmTrainReq = it->second;
        if ( fmTrainReq->isMasterReq() ) {
            mFMTrainReqMap.erase( it++ );
            delete fmTrainReq;
            continue;
        }
        ++it;
    }
}

void
DcgmFMLWLinkMsgHndlr::dumpInfo(std::ostream *os)
{
    DcgmFMLWLinkReqBaseMap::iterator it = mFMTrainReqMap.begin();

    if ( it != mFMTrainReqMap.end() ) {
        (it->second)->dumpInfo( os );
    }
}

