
#include "logging.h"
#include "DcgmFMLWLinkReqConnTrain.h"
#include "DcgmFMLWLinkError.h"

DcgmFMLWLinkReqConnTrain::DcgmFMLWLinkReqConnTrain(lwswitch::fmMessage *pFmMessage,
                                                   FMConnInterface *ctrlConnIntf,
                                                   DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                                   DcgmLFMLWLinkDevRepo *linkDevRepo)
    :DcgmFMLWLinkReqConnBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo),
    mReqState(REQ_STATE_TRAIN_NEW_REQUEST)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain::DcgmFMLWLinkReqConnTrain\n" );

    uint32 trainType = getReqType();

    // set actual request handler based on the request type in google protobuf
    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_masterToOffCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_masterOffToSafeCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_masterSafeToHSCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_masterHSToSafeCallback;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_masterSafeToOffCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_slaveToOffCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_slaveOffToSafeCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_slaveSafeToHSCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_slaveHSToSafeCallback;
            break;
        }
        case lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_slaveSafeToOffCallback;
            break;
        }
        default: {
            mTrainHndlrCB = DcgmFMLWLinkReqConnTrain::_ilwalidTrainCallback;
            break;
        }
    }
}

DcgmFMLWLinkReqConnTrain::~DcgmFMLWLinkReqConnTrain()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain ~DcgmFMLWLinkReqConnTrain\n" );
}

bool
DcgmFMLWLinkReqConnTrain::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processNewMasterRequest\n" );
    bool bReqCompleted = false;

    // get exclusive access to our request object
    lock();

    if ( getMasterNodeId() == getSlaveNodeId() ) {
        // both endpoints of the connection is local. No need to co-ordinate 
        // with peer local fabric manager
        doSingleNodeConnTrainReq();
        bReqCompleted = true;
    } else {
        // multi-node training. Sync with peer LFM
        dcgmReturn_t ret = sendSlaveTrainReqMsg( getSlaveReqType() );
        if (ret != DCGM_ST_OK) {
            // unable to send message to slave FM. Complete the request as error
            setCompletionStatus( FM_LWLINK_ST_SLAVE_FM_SOCKET_ERR );
            sendRequestCompletion();
            bReqCompleted = true;
        } else {
            // request is send to slave FM. Wait for the response from slave side
            setLwrrentState( REQ_STATE_TRAIN_SLAVE_CONFIRMATION );
        }
    }
    unLock();
    return bReqCompleted;
}

bool
DcgmFMLWLinkReqConnTrain::processNewSlaveRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processNewSlaveRequest\n" );
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
DcgmFMLWLinkReqConnTrain::processRespConfirm(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processRespConfirm\n" );
    bool bReqCompleted = false;
    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkResponseMsg rspMsg= linkMsg.rspmsg();

    // we received a training confirmation from peer slave LFM for this request
    // proceed with master LFM's next action.

    // get exclusive access to our request object
    lock();

    if (getLwrrentState() != REQ_STATE_TRAIN_SLAVE_CONFIRMATION) {
        PRINT_WARNING( "", "Unexpected req state during slave FM link train req confirmation\n");
        // TODO: handle this with an error to master FM
    } 

    // complete the request as failed if the slave is not success
    if (rspMsg.status() != FM_LWLINK_ST_SUCCESS ) {
        setCompletionStatus( rspMsg.status() );
        sendRequestCompletion();
        bReqCompleted = true;            
    } else {
        // call the corresponding registered link train helper function
        mTrainHndlrCB( this, bReqCompleted );
    }

    unLock();
    return bReqCompleted;
}

bool
DcgmFMLWLinkReqConnTrain::processRespComplete(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processRespComplete\n" );
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
            PRINT_WARNING( "", "Unexpected req state during slave FM link train req confirmation\n");
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
DcgmFMLWLinkReqConnTrain::processSlaveSync(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processSlaveSync\n" );
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
DcgmFMLWLinkReqConnTrain::processMasterSync(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processMasterSync\n" );
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
DcgmFMLWLinkReqConnTrain::processReqTimeOut()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain processReqTimeOut\n" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
DcgmFMLWLinkReqConnTrain::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Link Train Req Information" << std::endl;
    *os << "\t\tReqState:  " << mReqState << std::endl;

    // append base request dump information
    DcgmFMLWLinkReqConnBase::dumpInfo( os );
}

void
DcgmFMLWLinkReqConnTrain::doSingleNodeConnTrainReq(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain doSingleNodeConnTrainReq\n" );
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

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTRANODE_CONN, &trainParam );
    // copy link state information
    setMasterLinkState( trainParam.srcEndState );
    setSlaveLinkState( trainParam.srcEndState );    

    // copy the result and finish the request
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(trainParam.status) );

    sendRequestCompletion();
}

void
DcgmFMLWLinkReqConnTrain::_masterOffToSafeCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    fmTrainReq->masterOffToSafeCallback( bReqCompleted );
}

void
DcgmFMLWLinkReqConnTrain::masterOffToSafeCallback(bool &bReqCompleted)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain masterOffToSafeCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_off_to_safe );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
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
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }
}

void
DcgmFMLWLinkReqConnTrain::_masterSafeToHSCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{
    fmTrainReq->masterSafeToHSCallback( bReqCompleted );
}

void
DcgmFMLWLinkReqConnTrain::masterSafeToHSCallback(bool &bReqCompleted)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain masterSafeToHSCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            doMasterSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
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
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }
}

void
DcgmFMLWLinkReqConnTrain::_masterToOffCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                               bool &bReqCompleted)
{
    //TODO
}

void
DcgmFMLWLinkReqConnTrain::_masterHSToSafeCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{
    fmTrainReq->masterHSToSafeCallback( bReqCompleted );
}

void
DcgmFMLWLinkReqConnTrain::masterHSToSafeCallback(bool &bReqCompleted)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain masterHSToSafeCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_SLAVE_CONFIRMATION: {
            // we were waiting for slave confirmation. move to next step locally
            // first set the main link state for high to safe transition
            doMasterMainlinkTrainIoctl( &linkParam, lwlink_train_link_active_to_swcfg );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );
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
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
                sendRequestCompletion();
                bReqCompleted = true;
            } else {
                // wait for slave's final response before finishing the request
                setLwrrentState( REQ_STATE_TRAIN_FINAL_SLAVE_RESP );
            }
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }

}

void
DcgmFMLWLinkReqConnTrain::_masterSafeToOffCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                   bool &bReqCompleted)
{
    // TODO
}

void
DcgmFMLWLinkReqConnTrain::_slaveOffToSafeCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{
    fmTrainReq->slaveOffToSafeCallback( bReqCompleted );
}

void
DcgmFMLWLinkReqConnTrain::slaveOffToSafeCallback(bool &bReqCompleted)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain slaveOffToSafeCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set sub-link state
            doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_off_to_safe );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
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
            setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );                
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }
}

void
DcgmFMLWLinkReqConnTrain::_slaveSafeToHSCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                 bool &bReqCompleted)
{
    fmTrainReq->slaveSafeToHSCallback( bReqCompleted );
}


void
DcgmFMLWLinkReqConnTrain::slaveSafeToHSCallback(bool &bReqCompleted)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain slaveSafeToHSCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set sub-link state
            doSlaveSublinkTrainIoctl( &subLinkParam, lwlink_train_sublink_safe_to_hs );
            if ( subLinkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
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
            setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }
}

void
DcgmFMLWLinkReqConnTrain::_slaveToOffCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                              bool &bReqCompleted)
{
    //TODO
}

void
DcgmFMLWLinkReqConnTrain::_slaveHSToSafeCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                 bool &bReqCompleted)
{ 
    fmTrainReq->slaveHSToSafeCallback( bReqCompleted );
}

void
DcgmFMLWLinkReqConnTrain::slaveHSToSafeCallback(bool &bReqCompleted)
{ 
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain slaveHSToSafeCallback\n" );
    lwlink_train_internode_conn_link linkParam;
    lwlink_train_internode_conn_sublink subLinkParam;

    switch ( getLwrrentState() ) {
        case REQ_STATE_TRAIN_NEW_REQUEST: {
            // set main link first for hs to safe transition
            doSlaveMainlinkTrainIoctl( &linkParam, lwlink_train_link_active_to_swcfg );
            if ( linkParam.status != LWL_SUCCESS ) {
                setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkParam.status) );
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
            setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(subLinkParam.status) );
            sendRequestCompletion();
            bReqCompleted = true;
            break;
        }
        default: {
            PRINT_WARNING( "", "Unexpected req state in master FM for OFF to Safe training req\n");
            break;
        }
    }
}

void
DcgmFMLWLinkReqConnTrain::_slaveSafeToOffCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                  bool &bReqCompleted)
{
    //TODO
}

void
DcgmFMLWLinkReqConnTrain::_ilwalidTrainCallback(DcgmFMLWLinkReqConnTrain *fmTrainReq,
                                                bool &bReqCompleted)
{
    // this shouldn't happen
    // ASSERT()
    PRINT_ERROR( "", "Invalid train request handler\n" );
}

int
DcgmFMLWLinkReqConnTrain::doSlaveSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                                   lwlink_sublink_train_type toLinkState)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain doSlaveSublinkTrainIoctl\n" );
    int ret;

    memset( subLinkParam, 0 , sizeof(*subLinkParam) );
    subLinkParam->trainTo = toLinkState;
    subLinkParam->isMasterEnd = false; // slave FM node is treated as non master end
    subLinkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getSlaveGpuId() );
    subLinkParam->localEndPoint.nodeId = getSlaveNodeId();
    subLinkParam->localEndPoint.linkIndex = getSlaveLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, subLinkParam );
    // copy the link state information
    setSlaveLinkState( subLinkParam->localEndState );
    return ret;
}

int
DcgmFMLWLinkReqConnTrain::doMasterSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                                    lwlink_sublink_train_type toLinkState)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain doMasterSublinkTrainIoctl\n" );
    int ret;

    memset( subLinkParam, 0 , sizeof(*subLinkParam) );
    subLinkParam->trainTo = toLinkState;
    subLinkParam->isMasterEnd = true; // master FM node is treated as master end
    subLinkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getMasterGpuId() );
    subLinkParam->localEndPoint.nodeId = getMasterNodeId();
    subLinkParam->localEndPoint.linkIndex = getMasterLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK, subLinkParam );
    // copy the link state information
    setMasterLinkState( subLinkParam->localEndState );
    return ret;
}

int
DcgmFMLWLinkReqConnTrain::doSlaveMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                                    lwlink_link_train_type toLinkState)
{   
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain doSlaveMainlinkTrainIoctl\n" );
    int ret;

    memset( linkParam, 0 , sizeof(*linkParam) );
    linkParam->trainTo = toLinkState;
    linkParam->isMasterEnd = false; // slave FM node is treated as non master end
    linkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getSlaveGpuId() );
    linkParam->localEndPoint.nodeId = getSlaveNodeId();
    linkParam->localEndPoint.linkIndex = getSlaveLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, linkParam );
    // copy the link state information
    setSlaveLinkState( linkParam->localEndState );
    return ret;
}


int
DcgmFMLWLinkReqConnTrain::doMasterMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                                     lwlink_link_train_type toLinkState)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnTrain doMasterMainlinkTrainIoctl\n" );
    int ret;

    memset( linkParam, 0 , sizeof(*linkParam) );
    linkParam->trainTo = toLinkState;
    linkParam->isMasterEnd = true; // master FM node is treated as master end
    linkParam->localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( getMasterGpuId() );
    linkParam->localEndPoint.nodeId = getMasterNodeId();
    linkParam->localEndPoint.linkIndex = getMasterLinkIndex();

    ret = mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK, linkParam );
    // copy the link state
    setMasterLinkState( linkParam->localEndState );
    return ret;
}
