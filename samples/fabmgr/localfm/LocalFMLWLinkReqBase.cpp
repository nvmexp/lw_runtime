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
#include <stdexcept>
#include <sstream>

#include "fm_log.h"
#include "LocalFMLWLinkReqBase.h"
#include "FMLWLinkError.h"


LocalFMLWLinkReqBase::LocalFMLWLinkReqBase(lwswitch::fmMessage *pFmMessage,
                                           FMConnInterface *ctrlConnIntf,
                                           LocalFMLWLinkDrvIntf *linkDrvIntf,
                                           LocalFMLWLinkDevRepo *linkDevRepo)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqBase::LocalFMLWLinkReqBase\n" );

    lwosInitializeCriticalSection( &mLock );

    if ( !pFmMessage->has_lwlinkmsg() ) {
        std::ostringstream ss;
        ss << "lwlink request base: received link training request message without required fields";
        FM_LOG_ERROR("%s", ss.str().c_str());
        mTrainReqId = 0;
        mCompStatus = FM_LWLINK_ST_BAD_ARGS;
    } else {
        lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
        mTrainReqId = linkMsg.trainreqid();
        mCompStatus = FM_LWLINK_ST_SUCCESS;
    }

    mCtrlConnIntf = ctrlConnIntf;
    mLWLinkDrvIntf = linkDrvIntf;
    mLWLinkDevRepo = linkDevRepo;

    // parse the incoming GPB message and create our local context
    mReqType = pFmMessage->type();
    mFmMsgHdrReqId = pFmMessage->requestid();
}

LocalFMLWLinkReqBase::~LocalFMLWLinkReqBase()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqBase::~LocalFMLWLinkReqBase\n" );
    lwosDeleteCriticalSection( &mLock );
}

void
LocalFMLWLinkReqBase::lock(void)
{
    lwosEnterCriticalSection( &mLock );
}

void 
LocalFMLWLinkReqBase::unLock(void)
{
    lwosLeaveCriticalSection( &mLock );
}

void
LocalFMLWLinkReqBase::dumpInfo(std::ostream *os)
{
    *os << "\t\tmTrainReqId:  " << mTrainReqId << std::endl;
}

bool
LocalFMLWLinkReqBase::defaultReqHandler(lwswitch::fmMessage *pFmMessage,
                                       std::string errorPrefix)
{
    bool bReqCompleted = true;
    std::string reqStr = lwswitch::FabricManagerMessageType_Name( pFmMessage->type() );

    FM_LOG_ERROR( "link training request type %d doesn't have any corresponding %s \n",
                pFmMessage->type(), errorPrefix.c_str() );

    return bReqCompleted;
}

bool
LocalFMLWLinkReqBase::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    bool bReqCompleted = true;
    std::string reqStr = lwswitch::FabricManagerMessageType_Name( pFmMessage->type() );

    FM_LOG_ERROR( "new link training request type %d is handled by default implementation \n",
                pFmMessage->type() );

    return bReqCompleted;
}

bool
LocalFMLWLinkReqBase::processNewSlaveRequest(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Request" );
}

bool
LocalFMLWLinkReqBase::processRespConfirm(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Confirm Response" );
}

bool
LocalFMLWLinkReqBase::processRespComplete(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Response Complete" );
}

bool
LocalFMLWLinkReqBase::processSlaveSync(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "slave FM Sync" );
}

bool
LocalFMLWLinkReqBase::processMasterSync(lwswitch::fmMessage *pFmMessage)
{
    return defaultReqHandler(pFmMessage, "Master FM Sync" );
}

bool
LocalFMLWLinkReqBase::processReqTimeOut()
{
    bool bReqCompleted = true;
    FM_LOG_ERROR( "link training request type timeout is handled by default handler\n" );
    return bReqCompleted;
}

LocalFMLWLinkReqConnBase::LocalFMLWLinkReqConnBase(lwswitch::fmMessage *pFmMessage,
                                                   FMConnInterface *ctrlConnIntf,
                                                   LocalFMLWLinkDrvIntf *linkDrvIntf,
                                                   LocalFMLWLinkDevRepo *linkDevRepo)
    :LocalFMLWLinkReqBase(pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase::LocalFMLWLinkReqConnBase\n" );

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
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_SUBLINK:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH_MAINLINK: {
                mMasterReq = true;
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

LocalFMLWLinkReqConnBase::~LocalFMLWLinkReqConnBase()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase::~LocalFMLWLinkReqConnBase\n" );
}

void
LocalFMLWLinkReqConnBase::sendRequestCompletion(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendRequestCompletion\n" );

    // send the response to appropriate FM (LFM or GFM)
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

    if ( isMasterReq() ) {
        generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE, getCompletionStatus() );
        // the request is completed and the response is to GFM.
        // So use the actual FM requestId used by GFM as it is tracked.
        pFmMessage->set_requestid( getFmMsgHdrReqId() );
        // send the message to the Global FM node
        retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );
    } else {
        // the response is to slaveFM. Let FM lower layer sendMessage choose requestId
        generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE, getCompletionStatus() );
        // send the message to the Master Node FM
        FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendRequestCompletion\n" );
        retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, false );
    }

    if ( retVal != FM_INT_ST_OK ) {
        // can't do much, just log an error
        FM_LOG_WARNING( "error while sending link training complete message to fabric manager\n" );
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}


void
LocalFMLWLinkReqConnBase::sendSlaveConfirmation(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendSlaveConfirmation\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

    // when the masterFM request for slave confirmation, it is tracking the
    // request. So send the confirmation with MasterFM's requestId
    pFmMessage->set_requestid( getFmMsgHdrReqId() );
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM, getCompletionStatus());
    // send the message to the Master FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // log error and clear our context
        FM_LOG_WARNING( "error while sending link training slave confirm message to Master FM" );
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
LocalFMLWLinkReqConnBase::sendSlaveSyncMsg(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendSlaveSyncMsg" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

    // set status to success as failures will go as complete and not 
    // masterSync and SlaveSync uses arbitrary FM msg requestId filled by FM lower layer
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_SLAVE_SYNC, FM_LWLINK_ST_SUCCESS );

    // slave sync message to the Master FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( getMasterNodeId(), pFmMessage, true );

    if ( retVal != FM_INT_ST_OK ) {
        // log error and clear our context
        FM_LOG_WARNING( "error while sending link training slave sync message to Master FM" );
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
LocalFMLWLinkReqConnBase::sendMasterSyncMsg(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendMasterSyncMsg" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

    // set status to success as failures will complete the request and don't
    // masterSync and SlaveSync uses arbitrary FM msg requestId filled by FM message lower layer    
    generateTrainRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_MASTER_SYNC, FM_LWLINK_ST_SUCCESS );

    // send sync msg to slave FM node.
    retVal = mCtrlConnIntf->SendMessageToLfm( getSlaveNodeId(), pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // log error and clear our context
        FM_LOG_WARNING( "error while sending link training master sync message to peer FM" );
        // TODO: Handle this error proper
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}


void LocalFMLWLinkReqConnBase::generateTrainRespMsg(lwswitch::fmMessage *pFmMessage,
                                                    lwswitch::FabricManagerMessageType msgType,
                                                    int respStatus)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase generateTrainRespMsg respStatus=%d requestId=%llu", respStatus, getTrainReqId() );

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

FMIntReturn_t
LocalFMLWLinkReqConnBase::sendSlaveTrainReqMsg(lwswitch::FabricManagerMessageType msgType)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase sendSlaveTrainReqMsg" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

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
LocalFMLWLinkReqConnBase::dumpInfo(std::ostream *os)
{
    LocalFMLWLinkReqBase::dumpInfo( os );

    *os << "\t\tMasterNodeId:  " << mMasterNodeId << std::endl;
    *os << "\t\tMasterGpuOrSwitchId:  " << mMasterGpuOrSwitchId << std::endl;
    *os << "\t\tMasterLinkIndex:  " << mMasterLinkIndex << std::endl;
    *os << "\t\tSlaveNodeId:  " << mSlaveNodeId << std::endl;
    *os << "\t\tSlaveGpuOrSwitchId:  " << mSlaveGpuOrSwitchId << std::endl;
    *os << "\t\tSlaveLinkIndex:  " << mSlaveLinkIndex << std::endl;
}
