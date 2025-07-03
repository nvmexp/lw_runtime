
#include <stdexcept>

#include "logging.h"
#include "DcgmFMLWLinkReqBase.h"
#include "DcgmFMLWLinkError.h"


DcgmFMLWLinkReqBase::DcgmFMLWLinkReqBase(lwswitch::fmMessage *pFmMessage,
                                         FMConnInterface *ctrlConnIntf,
                                         DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                         DcgmLFMLWLinkDevRepo *linkDevRepo)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqBase::DcgmFMLWLinkReqBase\n" );

    lwosInitializeCriticalSection( &mLock );

    if ( !pFmMessage->has_lwlinkmsg() ) {
        PRINT_ERROR("", "FMLWLinkReq: received link training request message without required fields");
        throw std::runtime_error("FMLWLinkReq: received link training request message without required fields");
    }

    mCtrlConnIntf = ctrlConnIntf;
    mLWLinkDrvIntf = linkDrvIntf;
    mLWLinkDevRepo = linkDevRepo;

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();

    // parse the incoming GPB message and create our local context
    mTrainReqId = linkMsg.trainreqid();
    mReqType = pFmMessage->type();
    mDcmgMsgReqId = pFmMessage->requestid();
    mCompStatus = FM_LWLINK_ST_SUCCESS;
}

DcgmFMLWLinkReqBase::~DcgmFMLWLinkReqBase()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqBase::~DcgmFMLWLinkReqBase\n" );
    lwosDeleteCriticalSection( &mLock );
}

void
DcgmFMLWLinkReqBase::lock(void)
{
    lwosEnterCriticalSection( &mLock );
}

void 
DcgmFMLWLinkReqBase::unLock(void)
{
    lwosLeaveCriticalSection( &mLock );
}

void
DcgmFMLWLinkReqBase::dumpInfo(std::ostream *os)
{
    *os << "\t\tmTrainReqId:  " << mTrainReqId << std::endl;
}

bool
DcgmFMLWLinkReqBase::defaultReqHandler(lwswitch::fmMessage *pFmMessage,
                                       std::string errorPrefix)
{
    bool bReqCompleted = true;
    std::string reqStr = lwswitch::FabricManagerMessageType_Name( pFmMessage->type() );

    PRINT_ERROR( "%d %s", "Link request type %d doesn't have any corresponding %s \n",
                pFmMessage->type(), errorPrefix.c_str() );

    return bReqCompleted;
}

bool
DcgmFMLWLinkReqBase::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted = true;
    std::string reqStr = lwswitch::FabricManagerMessageType_Name( pFmMessage->type() );

    PRINT_ERROR( "%d", "New link request type %d is handled by default implementation \n",
                pFmMessage->type() );

    return bReqCompleted;
}

bool
DcgmFMLWLinkReqBase::processNewSlaveRequest(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Request" );
}

bool
DcgmFMLWLinkReqBase::processRespConfirm(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Confirm Response" );
}

bool
DcgmFMLWLinkReqBase::processRespComplete(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Response Complete" );
}

bool
DcgmFMLWLinkReqBase::processSlaveSync(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Sync" );
}

bool
DcgmFMLWLinkReqBase::processMasterSync(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "Master FM Sync" );
}

bool
DcgmFMLWLinkReqBase::processReqTimeOut()
{
    bool bReqCompleted = true;
    PRINT_ERROR( "", "Link request type timeout is handled by default handler\n");
    return bReqCompleted;
}

DcgmFMLWLinkReqConnBase::DcgmFMLWLinkReqConnBase(lwswitch::fmMessage *pFmMessage,
                                                 FMConnInterface *ctrlConnIntf,
                                                 DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                                 DcgmLFMLWLinkDevRepo *linkDevRepo)
    :DcgmFMLWLinkReqBase(pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase::DcgmFMLWLinkReqConnBase\n" );

    // get endpoint information from incoming GPB message.
    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkRequestMsg reqMsg = linkMsg.reqmsg();
    lwswitch::lwlinkTrainConnReqMsg trainConnReq = reqMsg.conntrainreqmsg();
    lwswitch::lwlinkConnectionInfo connInfo = trainConnReq.conninfo();
    lwswitch::lwlinkEndPointInfo masterEnd = connInfo.masterend();
    lwswitch::lwlinkEndPointInfo slaveEnd = connInfo.slaveend();

    // parse the incoming GPB message and create our local context
    mMasterNodeId = masterEnd.nodeid();
    mMasterGpuOrSwitchId = masterEnd.gpuorswitchid();
    mMasterLinkIndex = masterEnd.linkindex();
    mSlaveNodeId = slaveEnd.nodeid();
    mSlaveGpuOrSwitchId = slaveEnd.gpuorswitchid();
    mSlaveLinkIndex = slaveEnd.linkindex();

    // set whether this is a mater or slave training request
    // and corresponding slave request
    mMasterReq = false;
    mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF;

    switch ( getReqType() ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_SWITCH_OFF: {
                mMasterReq = true;            
                mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_SWITCH_OFF;
                break;
            }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE: {
                mMasterReq = true;            
                mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE;
                break;
            }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH: {
                mMasterReq = true;            
                mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH;
                break;
            }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE: {
                mMasterReq = true;            
                mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE;
                break;
            }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF: {
                mMasterReq = true;
                mSlaveReqType = lwswitch::FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF;
                break;
        }
        default: {
                // this default case is to keep the compiler warning off
                // as we don't handle all the message types in this switch case
                break;
        }
    }

    memset( &mMasterLinkState, 0 , sizeof(mMasterLinkState) );
    memset( &mSlaveLinkState, 0 , sizeof(mSlaveLinkState) );
}

DcgmFMLWLinkReqConnBase::~DcgmFMLWLinkReqConnBase()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase::~DcgmFMLWLinkReqConnBase\n" );
}

void
DcgmFMLWLinkReqConnBase::sendRequestCompletion(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase sendRequestCompletion\n" );

    // send the response to appropriate FM (LFM or GFM)
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;

    if ( isMasterReq() ) {
        generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE, getCompletionStatus() );
        // the request is completed and the response is to GFM.
        // So use the actual dcgm requestId used by GFM as it is tracked.
        pFmMessage->set_requestid( getDcgmMsgReqId() );
        // send the message to the Global FM node
        retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );
    } else {
        // the response is to slaveFM. Let dcgm lower layer sendMessage choose requestId
        generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE, getCompletionStatus() );
        // send the message to the Master Node FM
        retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, false );
    }

    if ( retVal != DCGM_ST_OK ) {
        // can't do much, just log an error
        PRINT_WARNING("", "error while sending train complete message to Fabric Manager\n");
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}


void
DcgmFMLWLinkReqConnBase::sendSlaveConfirmation(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase sendSlaveConfirmation\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;

    // when the masterFM request for slave confirmation, it is tracking the
    // request. So send the confirmation with MasterFM's requestId
    pFmMessage->set_requestid( getDcgmMsgReqId() );
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM, getCompletionStatus());
    // send the message to the Master FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, false );

    if ( retVal != DCGM_ST_OK ) {
        // log error and clear our context
        PRINT_WARNING("", "error while sending train confirm message to Master FM\n");
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
DcgmFMLWLinkReqConnBase::sendSlaveSyncMsg(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase sendSlaveSyncMsg\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;

    // set status to success as failures will go as complete and not 
    // masterSync and SlaveSync uses arbitrary dcgm requestId filled by dcgm lower layer
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_SYNC, FM_LWLINK_ST_SUCCESS );

    // slave sync message to the Master FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, true );

    if ( retVal != DCGM_ST_OK ) {
        // log error and clear our context
        PRINT_WARNING("", "error while sending train confirm message to Master FM\n");
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
DcgmFMLWLinkReqConnBase::sendMasterSyncMsg(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase sendMasterSyncMsg\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;

    // set status to success as failures will complete the request and don't
    // masterSync and SlaveSync uses arbitrary dcgm requestId filled by dcgm lower layer    
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_MASTER_SYNC, FM_LWLINK_ST_SUCCESS );

    // send sync msg to slave FM node.
    retVal = mCtrlConnIntf->SendMessageToLfm( getSlaveNodeId(), pFmMessage, false );

    if ( retVal != DCGM_ST_OK ) {
        // log error and clear our context
        PRINT_WARNING("", "error while sending train confirm message to Master FM\n");
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}


void DcgmFMLWLinkReqConnBase::generateTrainRespMsg(lwswitch::fmMessage *pFmMessage,
                                                   lwswitch::FabricManagerMessageType msgType,
                                                   int respStatus)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase generateTrainRespMsg\n" );

    // create the link state info
    lwswitch::lwlinkStateInfo *masterState = new lwswitch::lwlinkStateInfo();    
    lwswitch::lwlinkStateInfo *slaveState = new lwswitch::lwlinkStateInfo();    
    masterState->set_linkmode( getMasterLinkMode() );
    masterState->set_txsublinkmode( getMasterTxSubLinkMode() );
    masterState->set_rxsublinkmode( getMasterRxSubLinkMode() );
    slaveState->set_linkmode( getSlaveLinkMode() );
    slaveState->set_txsublinkmode( getSlaveTxSubLinkMode() );
    slaveState->set_rxsublinkmode( getSlaveRxSubLinkMode() );

    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();
    lwswitch::lwlinkTrainConnRspMsg *trainRspMsg = new lwswitch::lwlinkTrainConnRspMsg();
    trainRspMsg->set_allocated_masterstate( masterState );
    trainRspMsg->set_allocated_slavestate( slaveState );
    rspMsg->set_allocated_conntrainrspmsg( trainRspMsg );

    rspMsg->set_status( respStatus );

    // create the final train request message
    lwswitch::lwlinkMsg *linkMsg = new lwswitch::lwlinkMsg();
    linkMsg->set_trainreqid( getTrainReqId() );
    linkMsg->set_allocated_rspmsg( rspMsg );

    // fill the fabric message
    pFmMessage->set_type( msgType );
    pFmMessage->set_allocated_lwlinkmsg( linkMsg );
}

dcgmReturn_t
DcgmFMLWLinkReqConnBase::sendSlaveTrainReqMsg(lwswitch::FabricManagerMessageType msgType)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConnBase sendSlaveTrainReqMsg\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;

    // create the master and slave link information
    lwswitch::lwlinkEndPointInfo *masterEnd = new lwswitch::lwlinkEndPointInfo();
    lwswitch::lwlinkEndPointInfo *slaveEnd = new lwswitch::lwlinkEndPointInfo();

    masterEnd->set_nodeid( getMasterNodeId() );
    masterEnd->set_gpuorswitchid( getMasterGpuId() );
    masterEnd->set_linkindex( getMasterLinkIndex() );
    slaveEnd->set_nodeid( getSlaveNodeId() );
    slaveEnd->set_gpuorswitchid( getSlaveGpuId() );
    slaveEnd->set_linkindex( getSlaveLinkIndex() );

    // create the link connection pair
    lwswitch::lwlinkConnectionInfo *connInfo = new lwswitch::lwlinkConnectionInfo();
    connInfo->set_allocated_masterend( masterEnd );
    connInfo->set_allocated_slaveend( slaveEnd );

    // create the actual train request
    lwswitch::lwlinkRequestMsg *reqMsg = new lwswitch::lwlinkRequestMsg();
    lwswitch::lwlinkTrainConnReqMsg *trainReqMsg = new lwswitch::lwlinkTrainConnReqMsg();
    trainReqMsg->set_allocated_conninfo( connInfo );
    reqMsg->set_allocated_conntrainreqmsg( trainReqMsg );

    // create the final train request message
    lwswitch::lwlinkMsg *linkMsg = new lwswitch::lwlinkMsg();
    linkMsg->set_trainreqid( getTrainReqId() );
    linkMsg->set_allocated_reqmsg( reqMsg );

    // fill the fabric message
    pFmMessage->set_type( msgType );
    pFmMessage->set_allocated_lwlinkmsg( linkMsg );

    // send the message to the slave FM node
    // the slave confirm request is tracked so that master won't 
    // wait and keep the request in its context
    retVal = mCtrlConnIntf->SendMessageToLfm( getSlaveNodeId(), pFmMessage, false );

    // free the allocated  message and return final status
    delete( pFmMessage );
    return retVal;
}

void
DcgmFMLWLinkReqConnBase::dumpInfo(std::ostream *os)
{
    DcgmFMLWLinkReqBase::dumpInfo( os );

    *os << "\t\tMasterNodeId:  " << mMasterNodeId << std::endl;
    *os << "\t\tMasterGpuOrSwitchId:  " << mMasterGpuOrSwitchId << std::endl;
    *os << "\t\tMasterLinkIndex:  " << mMasterLinkIndex << std::endl;
    *os << "\t\tSlaveNodeId:  " << mSlaveNodeId << std::endl;
    *os << "\t\tSlaveGpuOrSwitchId:  " << mSlaveGpuOrSwitchId << std::endl;
    *os << "\t\tSlaveLinkIndex:  " << mSlaveLinkIndex << std::endl;
}
