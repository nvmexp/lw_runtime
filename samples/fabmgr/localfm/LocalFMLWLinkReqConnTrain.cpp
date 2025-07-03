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
#include "fm_log.h"
#include "LocalFMLWLinkReqConnTrain.h"
#include "FMLWLinkError.h"
#include <climits>

LocalFMLWLinkReqConnTrain::LocalFMLWLinkReqConnTrain(lwswitch::fmMessage *pFmMessage,
                                                     FMConnInterface *ctrlConnIntf,
                                                     LocalFMLWLinkDrvIntf *linkDrvIntf,
                                                     LocalFMLWLinkDevRepo *linkDevRepo)
    :LocalFMLWLinkReqConnBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo),
    mReqState(REQ_STATE_TRAIN_NEW_REQUEST)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain::LocalFMLWLinkReqConnTrain" );

    uint32 trainType = getReqType();
    mSkipSublinkTraining = false;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // set actual request handler based on the request type in google protobuf
    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterToOffCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterOffToSafeCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterSafeToHSCallback;
            break;
        }
        //TODO: add handlers for sublink and mainlink training for other states linke HIGH_TO_SAFE, SAFE_TO_OFF 
        // when adding the requests for those
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterSafeToHSNoPeerCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterHSToSafeCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_masterSafeToOffCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_slaveToOffCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_slaveOffToSafeCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_slaveSafeToHSCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_slaveHSToSafeCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_slaveSafeToOffCallback;
            break;
        }
        default: {
            mTrainHndlrCB = LocalFMLWLinkReqConnTrain::_ilwalidTrainCallback;
            break;
        }
    }
#endif
}

LocalFMLWLinkReqConnTrain::~LocalFMLWLinkReqConnTrain()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain ~LocalFMLWLinkReqConnTrain" );
}

bool
LocalFMLWLinkReqConnTrain::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processNewMasterRequest" );
    bool bReqCompleted = false;

    // get exclusive access to our request object
    lock();

    if ( getMasterNodeId() == getSlaveNodeId() ) {
        // both endpoints of the connection is local. No need to co-ordinate 
        // with peer local fabric manager
        doNodeConnTrainReq();
        bReqCompleted = true;
    } else {
        uint32 trainType = pFmMessage->type();

        // TODO: this should be applicable for other state transitions when 
        // peer to peer LFM communication is not ilwolved
        if (trainType == lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK ||
            trainType == lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK)
        {
            mTrainHndlrCB(this, bReqCompleted);
            unLock();
            return bReqCompleted;
        }

        // handle initoptimize to HS
        if ( getLwrrentState() == REQ_STATE_TRAIN_SLAVE_SUB_STATE) {
            sendMasterSyncMsg();
        } else {
            // multi-node training. Sync with peer LFM
            FMIntReturn_t ret = sendSlaveTrainReqMsg( getSlaveReqType() );
            if (ret != FM_INT_ST_OK) {
                // unable to send message to slave FM. Complete the request as error
                setCompletionStatus( FM_LWLINK_ST_SLAVE_FM_SOCKET_ERR );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // request is send to slave FM. Wait for the response from slave side
                setLwrrentState( REQ_STATE_TRAIN_SLAVE_CONFIRMATION );
            }
        }
    }
    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processNewSlaveRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processNewSlaveRequest" );
    bool bReqCompleted = false;

    // we received a training request from master LFM. This node is slave LFM
    // and it is supposed to confirm the request to master.

    // get exclusive access to our request object
    lock();

    // first send the slave confirmation to master LFM

    // NOTE - At this time slave LFM send confirmation always. Add condition
    // here if something is prohibiting from slave to proceed with training.
    sendSlaveConfirmation( );

    // procced with slave's next phase, which will issue the 
    // corresponding link state local ioctl
    mTrainHndlrCB( this, bReqCompleted );

    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processRespConfirm(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted = false;
    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processRespConfirm requestId=%lu status = %d", linkMsg.trainreqid(), rspMsg.status() );

    // we received a training confirmation from peer slave LFM for this request
    // proceed with master LFM's next action.

    // get exclusive access to our request object
    lock();

    if (getLwrrentState() != REQ_STATE_TRAIN_SLAVE_CONFIRMATION) {
        FM_LOG_WARNING( "unexpected request state during slave FM link train request confirmation" );
        // TODO: handle this with an error to master FM
    } 

    // complete the request as failed if the slave is not success
    if (rspMsg.status() != FM_LWLINK_ST_SUCCESS ) {
        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processRespConfirm: slave fail" );
        setCompletionStatus( rspMsg.status() );
        sendRequestCompletion();
        bReqCompleted = true;            
    } else {
        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processRespConfirm: slave success" );
        // call the corresponding registered link train helper function
        mTrainHndlrCB( this, bReqCompleted );
    }

    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processRespComplete(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processRespComplete" );
    bool bReqCompleted = false;
    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();

    // we received a training complete message from peer slave LFM for this request
    // proceed with master LFM's next action, which is to send the final status to
    // Global FM, who requested this connection training operation

    // get exclusive access to our request object
    lock();

    if (rspMsg.status() == FM_LWLINK_ST_SUCCESS ) {
        // master FM should be expecting this state
        // this shouldn't happen with invalid local state
        if (getLwrrentState() != REQ_STATE_TRAIN_FINAL_SLAVE_RESP) {
            FM_LOG_WARNING( "unexpected request state during slave FM link train request confirmation" );
        } 
    }

    // the request is completed. copy slave FM link state
    // Master LFM link state is already updated on the request
    lwlink_link_state state;
    lwswitch::lwlinkTrainConnRspMsg trainRspMsg = rspMsg.conntrainrspmsg();
    lwswitch::lwlinkStateInfo slaveState = trainRspMsg.slavestate();
    state.linkMode = slaveState.linkmode();
    state.txSubLinkMode = slaveState.txsublinkmode();
    state.rxSubLinkMode = slaveState.rxsublinkmode();
    setSlaveLinkState( state );

    // set the final status based on slave FM status
    if (rspMsg.status() != FM_LWLINK_ST_SUCCESS ) {
        setCompletionStatus( rspMsg.status() );
    } else {
        setCompletionStatus( FM_LWLINK_ST_SUCCESS );
    }

    // complete the request by responding to master FM
    sendRequestCompletion();
    bReqCompleted = true;

    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processSlaveSync(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processSlaveSync" );
    bool bReqCompleted = false;
    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();

    // we received a slave sync from peer slave LFM for this request
    // proceed with master LFM's next action.

    // get exclusive access to our request object
    lock();

    // copy slave FM link state
    lwlink_link_state state;
    lwswitch::lwlinkTrainConnRspMsg trainRspMsg = rspMsg.conntrainrspmsg();
    lwswitch::lwlinkStateInfo slaveState = trainRspMsg.slavestate();
    state.linkMode = slaveState.linkmode();
    state.txSubLinkMode = slaveState.txsublinkmode();
    state.rxSubLinkMode = slaveState.rxsublinkmode();
    setSlaveLinkState( state );

    // complete the request as failed if the slave is not success
    if (rspMsg.status() !=FM_LWLINK_ST_SUCCESS ) {
        setCompletionStatus( rspMsg.status() );
        sendRequestCompletion();
        bReqCompleted = true;            
    } else {
        // slave indicated success, let the master continue with next phase.
        // call the corresponding registered link train helper function
        mTrainHndlrCB( this, bReqCompleted );
    }

    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processMasterSync(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processMasterSync" );
    bool bReqCompleted = false;

    // we received a master sync from peer master LFM for this request
    // proceed with slave LFM's next action.

    // get exclusive access to our request object
    lock();

    // master won't send failure to slave.
    // just procced with slave's next phase
    mTrainHndlrCB( this, bReqCompleted );

    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConnTrain::processReqTimeOut()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain processReqTimeOut" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
LocalFMLWLinkReqConnTrain::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Link Train Req Information" << std::endl;
    *os << "\t\tReqState:  " << mReqState << std::endl;

    // append base request dump information
    LocalFMLWLinkReqConnBase::dumpInfo( os );
}

void
LocalFMLWLinkReqConnTrain::doNodeConnTrainReq(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain doNodeConnTrainReq" );
    lwlink_conn_train_type  trainTo = lwlink_train_conn_to_off;
    uint32 trainType = getReqType();

    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF: {
            trainTo = lwlink_train_conn_to_off;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE: {
            trainTo = lwlink_train_conn_off_to_swcfg;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH: {
            trainTo = lwlink_train_conn_swcfg_to_active;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            trainTo = lwlink_train_conn_active_to_swcfg;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            trainTo = lwlink_train_conn_swcfg_to_off;
            break;
        }
    }

    lwlink_train_intranode_conn trainParam;
    memset( &trainParam, 0 , sizeof(trainParam) );
    trainParam.trainTo = trainTo;
    // fill source endpoint information/bdf
    trainParam.srcEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo(getMasterGpuId());
    trainParam.srcEndPoint.nodeId = getMasterNodeId();
    trainParam.srcEndPoint.linkIndex = getMasterLinkIndex();
    // fill destination endpoint information/bdf    
    trainParam.dstEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo(getSlaveGpuId());
    trainParam.dstEndPoint.nodeId = getSlaveNodeId();
    trainParam.dstEndPoint.linkIndex = getSlaveLinkIndex();

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTRANODE_CONN, &trainParam, sizeof(trainParam) );
    // copy link state information
    setMasterLinkState( trainParam.srcEndState );
    setSlaveLinkState( trainParam.dstEndState );    

    // copy the result and finish the request
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(trainParam.status) );

    sendRequestCompletion();
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConnTrain::_masterOffToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                    bool &bReqCompleted)
{
    fmTrainReq->masterOffToSafeCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::masterOffToSafeCallback(bool &bReqCompleted)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain masterOffToSafeCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_off_to_safe );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send master sync to slave.
                // wait for slave response (barriers) before going to next step
                sendMasterSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_SLAVE_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_SLAVE_SUB_STATE: {
            // sub-link training is completed on both end.
            // issue main link state locally
            doMasterMainlinkTrainIoctl( &linkParam, lwlink_train_link_off_to_swcfg );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in master FM for OFF to Safe training request" );
            break;
        }
    }
}

void
LocalFMLWLinkReqConnTrain::_masterSafeToHSCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    fmTrainReq->masterSafeToHSCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::masterSafeToHSCallback(bool &bReqCompleted)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain masterSafeToHSCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            if (!mSkipSublinkTraining)
                doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            if ( !mSkipSublinkTraining && subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send master sync to slave.
                // wait for slave response (barriers) before going to next step
                sendMasterSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_SLAVE_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_SLAVE_SUB_STATE: {
            // sub-link training is completed on both end.
            // issue main link state locally
            doMasterMainlinkTrainIoctl( &linkParam, lwlink_train_link_swcfg_to_active );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in master FM for Safe to HS training request" );
            break;
        }
    }
}

void
LocalFMLWLinkReqConnTrain::_masterToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                bool &bReqCompleted)
{
    // TODO
}

void
LocalFMLWLinkReqConnTrain::_masterHSToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    fmTrainReq->masterHSToSafeCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::masterHSToSafeCallback(bool &bReqCompleted)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain masterHSToSafeCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            // first set the main link state for high to safe transition
            doMasterMainlinkTrainIoctl( &linkParam, lwlink_train_link_active_to_swcfg );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send master sync to slave.
                // wait for slave response (barriers) before going to next step
                sendMasterSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_SLAVE_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_SLAVE_SUB_STATE: {
            // set sub-link state to SAFE
            doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_hs_to_safe );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in master FM for HS to Safe training request" );
            break;
        }
    }

}

void
LocalFMLWLinkReqConnTrain::_masterSafeToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                    bool &bReqCompleted)
{
    // TODO
}

void
LocalFMLWLinkReqConnTrain::_slaveOffToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    fmTrainReq->slaveOffToSafeCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::slaveOffToSafeCallback(bool &bReqCompleted)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain slaveOffToSafeCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set sub-link state
            doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_off_to_safe );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send slave sync state to master FM.
                // then wait for master response (barriers) before going to next step
                sendSlaveSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_MASTER_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_MASTER_SUB_STATE: {
            // issue main link state locally. this is the last state for slave
            doSlaveMainlinkTrainIoctl( &linkParam, lwlink_train_link_off_to_swcfg );
            setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );                
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            FM_LOG_WARNING( "Unexpected request state in slave FM for OFF to Safe training request" );
            break;
        }
    }
}

void
LocalFMLWLinkReqConnTrain::_slaveSafeToHSCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{
    fmTrainReq->slaveSafeToHSCallback( bReqCompleted );
}


void
LocalFMLWLinkReqConnTrain::slaveSafeToHSCallback(bool &bReqCompleted)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain slaveSafeToHSCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set sub-link state
            if (!mSkipSublinkTraining)
                doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            if ( !mSkipSublinkTraining && subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                FM_LOG_DEBUG("REQ_STATE_TRAIN_NEW_REQUEST status = %d translated status=%d\n", linkParam.status, FMLWLinkError::getLinkErrorCode(subLinkParam.status));
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send slave sync state to master FM.
                // then wait for master response (barriers) before going to next step
                sendSlaveSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_MASTER_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_MASTER_SUB_STATE: {
            // issue main link state locally. this is the last state for slave
            doSlaveMainlinkTrainIoctl( &linkParam, lwlink_train_link_swcfg_to_active );
            FM_LOG_DEBUG("REQ_STATE_TRAIN_MASTER_SUB_STATE status = %d translated status=%d\n", linkParam.status, FMLWLinkError::getLinkErrorCode(linkParam.status));
            setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in salve FM for Safe to HS training request" );
            break;
        }
    }
}

void
LocalFMLWLinkReqConnTrain::_slaveToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                               bool &bReqCompleted)
{
    // TODO
}

void
LocalFMLWLinkReqConnTrain::_slaveHSToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{ 
    fmTrainReq->slaveHSToSafeCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::slaveHSToSafeCallback(bool &bReqCompleted)
{ 
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain slaveHSToSafeCallback" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set main link first for hs to safe transition
            doSlaveMainlinkTrainIoctl( &linkParam, lwlink_train_link_active_to_swcfg );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send slave sync state to master FM.
                // then wait for master response (barriers) before going to next step
                sendSlaveSyncMsg();
                setLwrrentState( REQ_STATE_TRAIN_MASTER_SUB_STATE );
            }
            break;
        }
        case REQ_STATE_TRAIN_MASTER_SUB_STATE: {
            // issue slave link state transition locally. this is the last state for slave
            doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_hs_to_safe );
            setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in salve FM for HS to Safe training request" );
            break;
        }
    }
}

void
LocalFMLWLinkReqConnTrain::_slaveSafeToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    // TODO
}

void
LocalFMLWLinkReqConnTrain::_ilwalidTrainCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                 bool &bReqCompleted)
{
    // this shouldn't happen
    // ASSERT()
    FM_LOG_ERROR( "invalid LWLink training request handler is called" );
}

void
LocalFMLWLinkReqConnTrain::_masterSafeToHSNoPeerCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                                         bool &bReqCompleted)
{
    fmTrainReq->masterSafeToHSNoPeerCallback( bReqCompleted );
}

void
LocalFMLWLinkReqConnTrain::masterSafeToHSNoPeerCallback(bool &bReqCompleted)
{
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getReqType() ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK: {
            // set sub-link state
            if (getMasterNodeId() == INT_MAX) {
                doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            } else {
                doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            }

            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                FM_LOG_DEBUG("REQ_STATE_TRAIN_NEW_REQUEST status = %d translated status=%d\n", linkParam.status, FMLWLinkError::getLinkErrorCode(subLinkParam.status));
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // send slave sync state to master FM.
                // then wait for master response (barriers) before going to next step
                setCompletionStatus( FMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            }
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK: {
            if (getMasterNodeId() == INT_MAX) {
                doSlaveMainlinkTrainIoctl( &linkParam, lwlink_train_link_swcfg_to_active );
            } else {
                doMasterMainlinkTrainIoctl( &linkParam, lwlink_train_link_swcfg_to_active );
            }
            FM_LOG_DEBUG("REQ_STATE_TRAIN_MASTER_SUB_STATE status = %d translated status=%d\n", linkParam.status, FMLWLinkError::getLinkErrorCode(linkParam.status));
            setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkParam.status) );
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            FM_LOG_WARNING( "unexpected request state in salve FM for Safe to HS training request" );
            break;
        }
    }
}

int
LocalFMLWLinkReqConnTrain::doSlaveSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                                   lwlink_sublink_train_type toLinkState)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain doSlaveSublinkTrainIoctl" );
    int ret;

    memset( subLinkParam, 0 , sizeof(*subLinkParam) );
    subLinkParam->trainTo = toLinkState;
    subLinkParam->isMasterEnd = false; // slave FM node is treated as non master end
    subLinkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getSlaveGpuId() );
    subLinkParam->localEndPoint.nodeId = getSlaveNodeId();
    subLinkParam->localEndPoint.linkIndex = getSlaveLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, subLinkParam, sizeof(*subLinkParam) );
    // copy the link state information
    setSlaveLinkState( subLinkParam->localEndState );
    return ret;
}

int
LocalFMLWLinkReqConnTrain::doMasterSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                                     lwlink_sublink_train_type toLinkState)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain doMasterSublinkTrainIoctl" );
    int ret;

    memset( subLinkParam, 0 , sizeof(*subLinkParam) );
    subLinkParam->trainTo = toLinkState;
    subLinkParam->isMasterEnd = true; // master FM node is treated as master end
    subLinkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getMasterGpuId() );
    subLinkParam->localEndPoint.nodeId = getMasterNodeId();
    subLinkParam->localEndPoint.linkIndex = getMasterLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, subLinkParam, sizeof(*subLinkParam) );
    // copy the link state information
    setMasterLinkState( subLinkParam->localEndState );
    return ret;
}

int
LocalFMLWLinkReqConnTrain::doSlaveMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                                     lwlink_link_train_type toLinkState)
{   
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain doSlaveMainlinkTrainIoctl" );
    int ret;

    memset( linkParam, 0 , sizeof(*linkParam) );
    linkParam->trainTo = toLinkState;
    linkParam->isMasterEnd = false; // slave FM node is treated as non master end
    linkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getSlaveGpuId() );
    linkParam->localEndPoint.nodeId = getSlaveNodeId();
    linkParam->localEndPoint.linkIndex = getSlaveLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, linkParam, sizeof(*linkParam) );
    // copy the link state information
    setSlaveLinkState( linkParam->localEndState );
    return ret;
}


int
LocalFMLWLinkReqConnTrain::doMasterMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                                      lwlink_link_train_type toLinkState)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrain doMasterMainlinkTrainIoctl" );
    int ret;

    memset( linkParam, 0 , sizeof(*linkParam) );
    linkParam->trainTo = toLinkState;
    linkParam->isMasterEnd = true; // master FM node is treated as master end
    linkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getMasterGpuId() );
    linkParam->localEndPoint.nodeId = getMasterNodeId();
    linkParam->localEndPoint.linkIndex = getMasterLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, linkParam, sizeof(*linkParam) );
    // copy the link state
    setMasterLinkState( linkParam->localEndState );
    return ret;
}
#endif
