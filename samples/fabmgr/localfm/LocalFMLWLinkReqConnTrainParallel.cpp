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
#include "LocalFabricManager.h"
#include "LocalFMLWLinkReqConnTrainParallel.h"
#include "FMLWLinkError.h"
#include "FMCommonTypes.h"
#include "FMLWLinkTypes.h"

LocalFMLWLinkReqConnTrainParallel::LocalFMLWLinkReqConnTrainParallel(lwswitch::fmMessage *pFmMessage,
                                                                     FMConnInterface *ctrlConnIntf,
                                                                     LocalFMLWLinkDrvIntf *linkDrvIntf,
                                                                     LocalFMLWLinkDevRepo *linkDevRepo)
    : LocalFMLWLinkReqBase(pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo), 
      mReqState(REQ_STATE_TRAIN_NEW_REQUEST)
{

    uint32 trainType = getReqType();

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkRequestMsg reqMsg = linkMsg.reqmsg();
    lwswitch::lwlinkTrainParallelConnReqMsg trainParallelConnReq = reqMsg.conntrainparallelreqmsg();

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel::LocalFMLWLinkReqConnTrainParallel size = %u trainType = %d", trainParallelConnReq.conninfo_size(), trainType);
    for (int i = 0 ; i < trainParallelConnReq.conninfo_size(); i++ ) {
        linkInfo link;
        auto connInfo = trainParallelConnReq.conninfo(i);
        
        auto masterEnd = connInfo.masterend();
        link.masterEnd.nodeId = masterEnd.nodeid();
        link.masterEnd.gpuOrSwitchId = masterEnd.gpuorswitchid(); 
        link.masterEnd.linkIndex = masterEnd.linkindex();
        link.masterEnd.qualityInfo.eomLow = false; // default value
        memset(&link.masterEnd.linkState, 0 , sizeof(link.masterEnd.linkState));
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        memset(&link.masterEnd.fomValues, 0, sizeof(link.masterEnd.fomValues));
        memset(&link.masterEnd.gradingValues, 0, sizeof(link.masterEnd.gradingValues));
#endif

        auto slaveEnd = connInfo.slaveend();
        link.slaveEnd.nodeId = slaveEnd.nodeid();
        link.slaveEnd.gpuOrSwitchId = slaveEnd.gpuorswitchid(); 
        link.slaveEnd.linkIndex = slaveEnd.linkindex();
        link.slaveEnd.qualityInfo.eomLow = false; // default value
        memset(&link.slaveEnd.linkState, 0 , sizeof(link.slaveEnd.linkState));
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        memset(&link.slaveEnd.fomValues, 0, sizeof(link.slaveEnd.fomValues));
        memset(&link.slaveEnd.gradingValues, 0, sizeof(link.slaveEnd.gradingValues));
#endif

        mLwlinks.push_back(link);
    }
    mMasterReq = false;

    // set actual request handler based on the request type in google protobuf
    switch ( trainType ) {
        // master train 
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
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
        case lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
#endif
        {
            mMasterReq = true;
            break;
        }
        default: {
            FM_LOG_WARNING( "Unknown request type %d", trainType );
            break;
        }

    }
}

LocalFMLWLinkReqConnTrainParallel::~LocalFMLWLinkReqConnTrainParallel()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel ~LocalFMLWLinkReqConnTrainParallel" );
}

void
LocalFMLWLinkReqConnTrainParallel::sendRequestCompletion()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel sendRequestCompletion\n" );
    // send the response to appropriate FM (LFM or GFM)
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;

    generateTrainParallelRespMsg( pFmMessage, lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE, getCompletionStatus());
    // the request is completed and the response is to GFM.
    // So use the actual FM requestId used by GFM as it is tracked.
    pFmMessage->set_requestid( getFmMsgHdrReqId() );
    // send the message to the Global FM node
    retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // can't do much, just log an error
        FM_LOG_WARNING( "error while sending link training complete message to fabric manager\n" );
    }

    // free the allocated  message and return final status
    delete( pFmMessage );

}

bool
LocalFMLWLinkReqConnTrainParallel::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel::processNewMasterRequest" );
    bool bReqCompleted = false;

    // get exclusive access to our request object
    lock();

    // both endpoints of the connection is local. No need to co-ordinate 
    // with peer local fabric manager
    uint32_t trainType = pFmMessage->type();
    switch ( trainType ) {
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE:
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF:
            doNodeConnTrainParallelReq(pFmMessage);
            break;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE:
            doParallelInitoptimizeReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_POST_INITOPTIMIZE:
            doParallelPostInitoptimizeReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_INF_MODE:
            doParallelEnableInfModeReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_INITOPTIMIZE_TO_HIGH:
            doParallelInternodeToHighSpeedReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_TO_OFF:
            doParallelInternodeToOffReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_TX:
            doParallelEnableMaintenanceTxReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_MAINTENANCE_RX:
            doParallelEnableMaintenanceRxReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_INF_MODE:
            doParallelDisableInfModeReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_ENABLE_FORCE_EQ:
            doParallelEnableForceEqReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_DISABLE_FORCE_EQ:
            doParallelDisableForceEqReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_OPTICAL_CHECK_EOM_STATUS:
            doParallelCheckEomStatusReq(pFmMessage);
            break;
        case lwswitch::FM_LWLINK_CONN_TRAIN_INTERNODE_GET_GRADING_AND_FOM_VALUES:
            doParallelGetGradingAndFomValuesReq(pFmMessage);
            break;
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_INTERNODE_PARALLEL_GET_LINK_STATE:
            doParallelGetLinkStateReq(pFmMessage);
            break;
#endif
    }
    bReqCompleted = true;
    // TODO: multi-node training. Sync with peer LFM
    unLock();
    return bReqCompleted;
}

void
LocalFMLWLinkReqConnTrainParallel::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Link Train Req Information" << std::endl;
    *os << "\t\tReqState:  " << mReqState << std::endl;

    // append base request dump information
    LocalFMLWLinkReqBase::dumpInfo( os );
}

void
LocalFMLWLinkReqConnTrainParallel::generateTrainParallelRespMsg(lwswitch::fmMessage *pFmMessage,
                                                                lwswitch::FabricManagerMessageType msgType,
                                                                int respStatus) 
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnBase generateTrainParallelRespMsg" );
    lwswitch::lwlinkTrainParallelConnRspMsg *trainParallelRspMsg = new lwswitch::lwlinkTrainParallelConnRspMsg();

    for (unsigned int i = 0; i < mLwlinks.size(); i++) {
        // State and Quality Info
        lwswitch::lwlinkStateInfo *masterState = new lwswitch::lwlinkStateInfo();
        masterState->set_linkmode( mLwlinks[i].masterEnd.linkState.linkMode);
        masterState->set_txsublinkmode( mLwlinks[i].masterEnd.linkState.txSubLinkMode );
        masterState->set_rxsublinkmode( mLwlinks[i].masterEnd.linkState.rxSubLinkMode );

        lwswitch::lwlinkQualityInfo *masterQualityInfo = new lwswitch::lwlinkQualityInfo();
        masterQualityInfo->set_eomlow( mLwlinks[i].masterEnd.qualityInfo.eomLow );
        masterState->set_allocated_qualityinfo(masterQualityInfo);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        // Fom Values
        lwswitch::lwlinkFomValues *fomValues = new lwswitch::lwlinkFomValues();
        fomValues->set_numlanes(mLwlinks[i].masterEnd.fomValues.numLanes);
        for(int j = 0; j < mLwlinks[i].masterEnd.fomValues.numLanes; j++)
            fomValues->add_fomvalues(mLwlinks[i].masterEnd.fomValues.fomValues[j]);
        masterState->set_allocated_fomvalues( fomValues );
        
        // GradingValues
        lwswitch::lwlinkGradingValues *gradingValues = new lwswitch::lwlinkGradingValues();
        gradingValues->set_lanemask(mLwlinks[i].masterEnd.gradingValues.laneMask);
        for(int j = 0; j < LWSWITCH_CCI_XVCR_LANES; j++) {
            gradingValues->add_txinit(mLwlinks[i].masterEnd.gradingValues.txInit[j]);
            gradingValues->add_rxinit(mLwlinks[i].masterEnd.gradingValues.rxInit[j]);
            gradingValues->add_txmaint(mLwlinks[i].masterEnd.gradingValues.txMaint[j]);
            gradingValues->add_rxmaint(mLwlinks[i].masterEnd.gradingValues.rxMaint[j]);
        }
        masterState->set_allocated_gradingvalues( gradingValues );
#endif

        lwswitch::lwlinkStateInfo *slaveState = new lwswitch::lwlinkStateInfo();
        slaveState->set_linkmode( mLwlinks[i].slaveEnd.linkState.linkMode );
        slaveState->set_txsublinkmode( mLwlinks[i].slaveEnd.linkState.txSubLinkMode);
        slaveState->set_rxsublinkmode( mLwlinks[i].slaveEnd.linkState.rxSubLinkMode);

        lwswitch::lwlinkQualityInfo *slaveQualityInfo = new lwswitch::lwlinkQualityInfo();
        slaveQualityInfo->set_eomlow( mLwlinks[i].masterEnd.qualityInfo.eomLow );
        slaveState->set_allocated_qualityinfo(slaveQualityInfo);



        lwswitch::lwlinkEndPointInfo *masterEnd = new lwswitch::lwlinkEndPointInfo();
        masterEnd->set_nodeid( mLwlinks[i].masterEnd.nodeId );
        masterEnd->set_gpuorswitchid(mLwlinks[i].masterEnd.gpuOrSwitchId);
        masterEnd->set_linkindex(mLwlinks[i].masterEnd.linkIndex);
        masterEnd->set_allocated_state(masterState);

        lwswitch::lwlinkEndPointInfo *slaveEnd = new lwswitch::lwlinkEndPointInfo();
        slaveEnd->set_nodeid( mLwlinks[i].slaveEnd.nodeId );
        slaveEnd->set_gpuorswitchid(mLwlinks[i].slaveEnd.gpuOrSwitchId);
        slaveEnd->set_linkindex(mLwlinks[i].slaveEnd.linkIndex);
        slaveEnd->set_allocated_state(slaveState);

        lwswitch::lwlinkConnectionInfo *connRspInfo = trainParallelRspMsg->add_connrspinfo();

        connRspInfo->set_allocated_masterend(masterEnd);
        connRspInfo->set_allocated_slaveend(slaveEnd);
    }
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();
    rspMsg->set_allocated_conntrainparallelrspmsg( trainParallelRspMsg );
    rspMsg->set_status( respStatus );

    lwswitch::lwlinkMsg *linkMsg = new lwswitch::lwlinkMsg();
    linkMsg->set_trainreqid( getTrainReqId() );
    linkMsg->set_allocated_rspmsg( rspMsg );

    // fill the fabric message
    pFmMessage->set_type( msgType );
    pFmMessage->set_allocated_lwlinkmsg( linkMsg );
}

void
LocalFMLWLinkReqConnTrainParallel::doNodeConnTrainParallelReq(lwswitch::fmMessage *pFmMessage)
{
    lwlink_conn_train_type  trainTo = lwlink_train_conn_to_off;
    uint32 trainType = getReqType();

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doNodeConnTrainParallelReq trainType = %d", trainType );
    switch ( trainType ) {
        // Parallel training requests
        case lwswitch::FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF: {
            trainTo = lwlink_train_conn_to_off;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE: {
            trainTo = lwlink_train_conn_off_to_swcfg;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH: {
            trainTo = lwlink_train_conn_swcfg_to_active;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE: {
            trainTo = lwlink_train_conn_active_to_swcfg;
            break;
        }
        case lwswitch::FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF: {
            trainTo = lwlink_train_conn_swcfg_to_off;
            break;
        }
    }

    //
    // lwlink_train_intranode_conns_parallel holds connection information for 288 LWLink connections.
    // This make the total size of this structure as 16140 bytes (~16KB). So, use this structure as a
    // dynamically allocated memory instead of stack variable.
    //
    lwlink_train_intranode_conns_parallel* pTrainParam = NULL;
    pTrainParam = (lwlink_train_intranode_conns_parallel*) calloc(1, sizeof(lwlink_train_intranode_conns_parallel));
    if (pTrainParam == NULL) {
        FM_LOG_ERROR("failed to allocate required memory to hold LWLink connections, marking LWLink training request as failed");
        setCompletionStatus( FMLWLinkError::getLinkErrorCode(FM_LWLINK_ST_LWL_NO_MEM) );
        sendRequestCompletion();
        return;
    }

    pTrainParam->trainTo = trainTo;
    pTrainParam->endPointPairsCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointPairsCount; i++) {
        pTrainParam->endPointPairs[i].src.nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPointPairs[i].src.linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPointPairs[i].src.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        pTrainParam->endPointPairs[i].dst.nodeId = mLwlinks[i].slaveEnd.nodeId;;
        pTrainParam->endPointPairs[i].dst.linkIndex = mLwlinks[i].slaveEnd.linkIndex;
        pTrainParam->endPointPairs[i].dst.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].slaveEnd.gpuOrSwitchId );
        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel training: %llu, %u <=======> %llu,%u" , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex, 
                      mLwlinks[i].slaveEnd.gpuOrSwitchId, mLwlinks[i].slaveEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTRANODE_CONNS_PARALLEL, pTrainParam, sizeof(lwlink_train_intranode_conns_parallel) );

    for(unsigned int i = 0; i < pTrainParam->endPointPairsCount; i++) {
        mLwlinks[i].masterEnd.linkState.linkMode = pTrainParam->endpointPairsStates[i].srcEnd.linkMode;
        mLwlinks[i].masterEnd.linkState.txSubLinkMode = pTrainParam->endpointPairsStates[i].srcEnd.txSubLinkMode;
        mLwlinks[i].masterEnd.linkState.rxSubLinkMode = pTrainParam->endpointPairsStates[i].srcEnd.rxSubLinkMode;

        mLwlinks[i].slaveEnd.linkState.linkMode = pTrainParam->endpointPairsStates[i].dstEnd.linkMode;
        mLwlinks[i].slaveEnd.linkState.txSubLinkMode = pTrainParam->endpointPairsStates[i].dstEnd.txSubLinkMode;
        mLwlinks[i].slaveEnd.linkState.rxSubLinkMode = pTrainParam->endpointPairsStates[i].dstEnd.rxSubLinkMode;
        FM_LOG_DEBUG("IOCTL link state (%d,%d,%d) <====> (%d,%d,%d)", pTrainParam->endpointPairsStates[i].srcEnd.linkMode,
                     pTrainParam->endpointPairsStates[i].srcEnd.txSubLinkMode, pTrainParam->endpointPairsStates[i].srcEnd.rxSubLinkMode, 
                     pTrainParam->endpointPairsStates[i].dstEnd.linkMode, pTrainParam->endpointPairsStates[i].dstEnd.txSubLinkMode,
                     pTrainParam->endpointPairsStates[i].dstEnd.rxSubLinkMode);
    }
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    sendRequestCompletion();

    free(pTrainParam);
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConnTrainParallel::doParallelInitoptimizeReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelInitoptimizeReq ");

    lwlink_train_internode_links_initoptimize *pTrainParam = new(lwlink_train_internode_links_initoptimize);
    memset( pTrainParam, 0 , sizeof(lwlink_train_internode_links_initoptimize) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelInitoptimizeReq size = %lu", mLwlinks.size());
    pTrainParam->endPointCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel training: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE, pTrainParam, sizeof(lwlink_train_internode_links_initoptimize) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG("IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_INITOPTIMIZE status=%d\n", pTrainParam->status);
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelPostInitoptimizeReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelPostInitoptimizeReq ");

    lwlink_train_internode_links_post_initoptimize *pTrainParam = new(lwlink_train_internode_links_post_initoptimize);
    memset( pTrainParam, 0 , sizeof(lwlink_train_internode_links_post_initoptimize) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelPostInitoptimizeReq size = %lu", mLwlinks.size());
    pTrainParam->endPointCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel training: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE, pTrainParam, sizeof(lwlink_train_internode_links_post_initoptimize) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG("IOCTL_LWLINK_TRAIN_INTERNODE_LINKS_POST_INITOPTIMIZE status=%d\n", pTrainParam->status);
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelEnableInfModeReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelEnableInfModeReq ");

    lwlink_optical_set_infinite_mode *pTrainParam = new(lwlink_optical_set_infinite_mode);
    memset( pTrainParam, 0 , sizeof(lwlink_optical_set_infinite_mode) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel EnableInfModeReq size = %lu", mLwlinks.size());
    pTrainParam->endPointCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel enable infinite mode: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_ENABLE_INFINITE_MODE, pTrainParam, sizeof(lwlink_optical_set_infinite_mode) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_ENABLE_INFINITE_MODE status=%d\n", pTrainParam->status);
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelEnableMaintenanceReq(lwswitch::fmMessage *pFmMessage, bool isTx)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelEnableMaintenanceReq Tx=%d", isTx);

    lwlink_optical_enable_maintenance *pTrainParam = new(lwlink_optical_enable_maintenance);
    memset( pTrainParam, 0 , sizeof(lwlink_optical_enable_maintenance) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel EnableMaintenanceReq size = %lu", mLwlinks.size());
    pTrainParam->bTx = isTx;
    pTrainParam->endPointCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel enable maintenance: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_ENABLE_MAINTENANCE, pTrainParam, sizeof(lwlink_optical_enable_maintenance) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_ENABLE_MAINTENANCE status=%d\n", pTrainParam->status);
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelEnableMaintenanceTxReq(lwswitch::fmMessage *pFmMessage)
{
    doParallelEnableMaintenanceReq(pFmMessage, true);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelEnableMaintenanceRxReq(lwswitch::fmMessage *pFmMessage)
{
    doParallelEnableMaintenanceReq(pFmMessage, false);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelDisableInfModeReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelDisableInfModeReq ");

    lwlink_optical_set_infinite_mode *pTrainParam = new(lwlink_optical_set_infinite_mode);
    memset( pTrainParam, 0 , sizeof(lwlink_optical_set_infinite_mode) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel DisableInfModeReq size = %lu", mLwlinks.size());
    pTrainParam->endPointCount = mLwlinks.size();
    for(unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel disable infinite mode: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_DISABLE_INFINITE_MODE, pTrainParam, sizeof(lwlink_optical_set_infinite_mode) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_DISABLE_INFINITE_MODE status=%d\n", pTrainParam->status);
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelInternodeToHighSpeedReq( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelInternodeToHighSpeedReq " );

    lwlink_train_internode_conns_parallel *pTrainParam = new(lwlink_train_internode_conns_parallel);
    memset( pTrainParam, 0 , sizeof(lwlink_train_internode_conns_parallel) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelInternodeToHighSpeedReq size = %lu", mLwlinks.size() );
    pTrainParam->localEndPointCount = mLwlinks.size();
    pTrainParam->trainTo = lwlink_train_link_swcfg_to_active;
    // The info in mLwlinks[i].slaveEnd end is ignored
    for( unsigned int i = 0; i < pTrainParam->localEndPointCount; i++ ) {
        pTrainParam->localEndPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->localEndPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->localEndPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        // For each link mLwlinks[i].slaveEnd.nodeId is set to true if it is the master end
        // otherwise mLwlinks[i].slaveEnd.nodeId is set to false
        if ( mLwlinks[i].slaveEnd.nodeId == true )
            pTrainParam->isMasterEnd[i] = true;
        else
            pTrainParam->isMasterEnd[i] = false;


        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel training: (nodeId, bus, linkIndex)  %u, %u, %u IsMaster %u" , 
                      pTrainParam->localEndPoints[i].nodeId, pTrainParam->localEndPoints[i].pciInfo.bus , pTrainParam->localEndPoints[i].linkIndex, 
                      mLwlinks[i].slaveEnd.nodeId );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL, pTrainParam, sizeof(lwlink_train_internode_conns_parallel) );

    for( unsigned int i = 0; i < pTrainParam->localEndPointCount; i++ ) {
        // only local end point is used so we always fill it in master
        mLwlinks[i].masterEnd.linkState.linkMode = pTrainParam->localEndStates[i].linkMode;
        mLwlinks[i].masterEnd.linkState.txSubLinkMode = pTrainParam->localEndStates[i].txSubLinkMode;
        mLwlinks[i].masterEnd.linkState.rxSubLinkMode = pTrainParam->localEndStates[i].rxSubLinkMode;

        FM_LOG_DEBUG( "IOCTL (nodeId, linkIndex, busId) = (%u, %u, %u)", pTrainParam->localEndPoints[i].nodeId,
                      pTrainParam->localEndPoints[i].linkIndex, pTrainParam->localEndPoints[i].pciInfo.bus );
        FM_LOG_DEBUG( "IOCTL link state (%d,%d,%d)", pTrainParam->localEndStates[i].linkMode,
                      pTrainParam->localEndStates[i].txSubLinkMode, pTrainParam->localEndStates[i].rxSubLinkMode );
    }
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    sendRequestCompletion();
    delete(pTrainParam);

    FM_LOG_DEBUG( "IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL status=%d\n", pTrainParam->status );
}
void
LocalFMLWLinkReqConnTrainParallel::doParallelInternodeToOffReq( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG( "Entering " );

    lwlink_train_internode_conns_parallel *pTrainParam = new(lwlink_train_internode_conns_parallel);
    memset( pTrainParam, 0 , sizeof(lwlink_train_internode_conns_parallel) );

    FM_LOG_DEBUG( "size = %lu", mLwlinks.size() );
    pTrainParam->localEndPointCount = mLwlinks.size();
    pTrainParam->trainTo = lwlink_train_conn_to_off;
    // The info in mLwlinks[i].slaveEnd end is ignored
    for( unsigned int i = 0; i < pTrainParam->localEndPointCount; i++ ) {
        pTrainParam->localEndPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->localEndPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->localEndPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        // For each link mLwlinks[i].slaveEnd.nodeId is set to true if it is the master end
        // otherwise mLwlinks[i].slaveEnd.nodeId is set to false
        if ( mLwlinks[i].slaveEnd.nodeId == true )
            pTrainParam->isMasterEnd[i] = true;
        else
            pTrainParam->isMasterEnd[i] = false;


        FM_LOG_DEBUG( "Parallel training to off: (nodeId, bus, linkIndex)  %u, %u, %u IsMaster %u" , 
                      pTrainParam->localEndPoints[i].nodeId, pTrainParam->localEndPoints[i].pciInfo.bus , pTrainParam->localEndPoints[i].linkIndex, 
                      mLwlinks[i].slaveEnd.nodeId );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL, pTrainParam, sizeof(lwlink_train_internode_conns_parallel) );

    for( unsigned int i = 0; i < pTrainParam->localEndPointCount; i++ ) {
        // only local end point is used so we always fill it in master
        mLwlinks[i].masterEnd.linkState.linkMode = pTrainParam->localEndStates[i].linkMode;
        mLwlinks[i].masterEnd.linkState.txSubLinkMode = pTrainParam->localEndStates[i].txSubLinkMode;
        mLwlinks[i].masterEnd.linkState.rxSubLinkMode = pTrainParam->localEndStates[i].rxSubLinkMode;

        FM_LOG_DEBUG( "IOCTL (nodeId, linkIndex, busId) = (%u, %u, %u)", pTrainParam->localEndPoints[i].nodeId,
                      pTrainParam->localEndPoints[i].linkIndex, pTrainParam->localEndPoints[i].pciInfo.bus );
        FM_LOG_DEBUG( "IOCTL link state (%d,%d,%d)", pTrainParam->localEndStates[i].linkMode,
                      pTrainParam->localEndStates[i].txSubLinkMode, pTrainParam->localEndStates[i].rxSubLinkMode );
    }
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    sendRequestCompletion();
    delete(pTrainParam);

    FM_LOG_DEBUG( "IOCTL_LWLINK_TRAIN_INTERNODE_CONNS_PARALLEL status=%d\n", pTrainParam->status );
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelEnableForceEqReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelEnableForceEqReq ");

    lwlink_optical_set_force_eq *pTrainParam = new(lwlink_optical_set_force_eq);
    memset( pTrainParam, 0 , sizeof(lwlink_optical_set_force_eq) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel EnableForceEqReq size = %lu", mLwlinks.size() );
    pTrainParam->endPointCount = mLwlinks.size();
    for (unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: enable force eq: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );
    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_ENABLE_FORCE_EQ, pTrainParam, sizeof(lwlink_optical_set_force_eq) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG( "IOCTL_LWLINK_OPTICAL_ENABLE_FORCE_EQ status=%d\n", pTrainParam->status );
    sendRequestCompletion();
    delete(pTrainParam);
}

void
LocalFMLWLinkReqConnTrainParallel::doParallelDisableForceEqReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelDisableForceEqReq ");

    lwlink_optical_set_force_eq *pTrainParam = new(lwlink_optical_set_force_eq);
    memset( pTrainParam, 0 , sizeof(lwlink_optical_set_force_eq) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel DisableForceEqReq size = %lu", mLwlinks.size() );
    pTrainParam->endPointCount = mLwlinks.size();
    for (unsigned int i = 0; i < pTrainParam->endPointCount; i++) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: disable force eq: %llu, %u " , 
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );
    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_DISABLE_FORCE_EQ, pTrainParam, sizeof(lwlink_optical_set_force_eq) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG( "IOCTL_LWLINK_OPTICAL_DISABLE_FORCE_EQ status=%d\n", pTrainParam->status );
    sendRequestCompletion();
    delete(pTrainParam);
}

bool
LocalFMLWLinkReqConnTrainParallel::getPhyIdFromDeviceId(FMGpuOrSwitchId_t deviceId, FMPhysicalId_t &physicalId)
{

    // get PCI from deviceId
    lwlink_pci_dev_info picInfo = mLWLinkDevRepo->getDevicePCIInfo( deviceId );

    // get phyId from PCI
    LocalFabricManagerControl *pLfmControl = (LocalFabricManagerControl*)mCtrlConnIntf;
    FMLWSwitchInfoList switchInfoList;
    pLfmControl->getAllLwswitchInfo(switchInfoList);

    for(FMLWSwitchInfoList::iterator it=switchInfoList.begin(); it!= switchInfoList.end(); it++)
    {
        FMLWSwitchInfo switchInfo;
        switchInfo = *it;
        if((switchInfo.pciInfo.domain == picInfo.domain) && (switchInfo.pciInfo.bus == picInfo.bus)&&
            (switchInfo.pciInfo.device == picInfo.device) && (switchInfo.pciInfo.function == picInfo.function) )
        {
            physicalId = switchInfo.physicalId;
            return true;
        }
    }
    return false;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConnTrainParallel::doParallelGetGradingAndFomValuesReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "Entering " );
    int status = FM_LWLINK_ST_SUCCESS;
    bool retVal;
    for(unsigned int i = 0; i < mLwlinks.size(); i++)
    {
        FMPhysicalId_t phyId;
        retVal = getPhyIdFromDeviceId(mLwlinks[i].masterEnd.gpuOrSwitchId, phyId);
        if(retVal == false)
        {
            FM_LOG_ERROR("request to get grading and FOM values failed as specified LWSwitch device not found");
            setCompletionStatus( FMLWLinkError::getLinkErrorCode(FM_LWLINK_ST_LWL_NOT_FOUND) );
            sendRequestCompletion();
            return;
        }
        // get switchInterface from phyId
        LocalFabricManagerControl *pLfmControl = (LocalFabricManagerControl*)mCtrlConnIntf;
        LocalFMSwitchInterface *pSwitchInterface;
        pSwitchInterface = pLfmControl->switchInterfaceAt( phyId );
        if ( !pSwitchInterface ) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object for physical id %d", phyId);

            setCompletionStatus( FMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND ) );
            sendRequestCompletion();
            return;
        }

        // Retrieve grading params
        LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS gradingParams;
        memset(&gradingParams, 0, sizeof(gradingParams));
        gradingParams.linkId = mLwlinks[i].masterEnd.linkIndex;

        switchIoctl_t ioctlStruct;
        memset(&ioctlStruct, 0, sizeof(ioctlStruct));
        ioctlStruct.type = IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES;
        ioctlStruct.ioctlParams = &gradingParams;
        ioctlStruct.paramSize = sizeof(gradingParams);
        FM_LOG_DEBUG( "calling IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES for switch physicalId:%d linkIndex:%d",\
                      phyId, gradingParams.linkId);
        FMIntReturn_t ret = pSwitchInterface->doIoctl( &ioctlStruct );
        if ( ret != FM_INT_ST_OK ) {
            FM_LOG_ERROR("failed to retrieve grading values for switch physicalId:%d linkIndex:%d with error:%d",\
                          phyId, gradingParams.linkId, ret);
            status = FMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
            break;
        }
        mLwlinks[i].masterEnd.gradingValues.laneMask = gradingParams.laneMask;
        memcpy(mLwlinks[i].masterEnd.gradingValues.txInit, gradingParams.grading.tx_init,
               sizeof(mLwlinks[i].masterEnd.gradingValues.txInit));
        memcpy(mLwlinks[i].masterEnd.gradingValues.rxInit, gradingParams.grading.rx_init,
               sizeof(mLwlinks[i].masterEnd.gradingValues.rxInit));
        memcpy(mLwlinks[i].masterEnd.gradingValues.txMaint, gradingParams.grading.tx_maint,
               sizeof(mLwlinks[i].masterEnd.gradingValues.txMaint));
        memcpy(mLwlinks[i].masterEnd.gradingValues.rxMaint, gradingParams.grading.rx_maint,
               sizeof(mLwlinks[i].masterEnd.gradingValues.rxMaint));

        // Retrieve FOM values
        LWSWITCH_GET_FOM_VALUES_PARAMS fomParams;
        memset(&fomParams, 0 ,sizeof(fomParams));
        fomParams.linkId = mLwlinks[i].masterEnd.linkIndex;

        ioctlStruct.type = IOCTL_LWSWITCH_GET_FOM_VALUES;
        ioctlStruct.ioctlParams = &fomParams;
        ioctlStruct.paramSize = sizeof(fomParams);
        FM_LOG_DEBUG( "calling IOCTL_LWSWITCH_GET_FOM_VALUES for switch physicalId:%d linkIndex:%d",\
                      phyId, fomParams.linkId);
        ret = pSwitchInterface->doIoctl( &ioctlStruct );
        if ( ret != FM_INT_ST_OK ) {
            FM_LOG_ERROR("failed to retrieve FOM values for switch physicalId:%d linkIndex:%d with error:%d",\
                          phyId, fomParams.linkId, ret);
            status = FMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
            break;
        }
        mLwlinks[i].masterEnd.fomValues.numLanes = fomParams.numLanes;
        memcpy(mLwlinks[i].masterEnd.fomValues.fomValues, fomParams.figureOfMeritValues,
               sizeof(mLwlinks[i].masterEnd.fomValues.fomValues));
    }
    
    setCompletionStatus( status );
    sendRequestCompletion();
    if( status == FM_LWLINK_ST_SUCCESS)
        FM_LOG_DEBUG( "Success" );
    else
        FM_LOG_ERROR( "Failure" );
}
#endif

void
LocalFMLWLinkReqConnTrainParallel::doParallelCheckEomStatusReq(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelCheckEomStatusReq " );

    lwlink_optical_check_eom_status *pTrainParam = new(lwlink_optical_check_eom_status);
    // reset ioctl parameters to ensures that EOM low value is set to false as this IOCTL is a no-op for non-optical links
    memset( pTrainParam, 0 , sizeof(lwlink_optical_check_eom_status) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelCheckEomStatusReq size = %lu", mLwlinks.size() );
    pTrainParam->endPointCount = mLwlinks.size();
    // The info in mLwlinks[i].slaveEnd end is ignored
    for ( unsigned int i = 0; i < pTrainParam->endPointCount; i++ ) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: get EOM status: %llu, %u " ,
                      mLwlinks[i].masterEnd.gpuOrSwitchId, mLwlinks[i].masterEnd.linkIndex );
    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_CHECK_EOM_STATUS, pTrainParam, sizeof(lwlink_optical_check_eom_status) );

    FM_LOG_DEBUG("LocalFMLWLinkReqConnTrainParallel doParallelCheckEomStatusReq: returned endpoint count = %u", pTrainParam->endPointCount);
    for ( unsigned int i = 0; i < pTrainParam->endPointCount; i++ ) {
        //
        // populate our link EOM state
        // only one end point is used so we always fill it in master
        // 
        mLwlinks[i].masterEnd.qualityInfo.eomLow = pTrainParam->bEomLow[i];

        FM_LOG_DEBUG( "IOCTL (busId, linkIndex, eomLow) = (%u, %u, %d)", pTrainParam->endPoints[i].pciInfo.bus,
                      pTrainParam->endPoints[i].linkIndex, pTrainParam->bEomLow[i] );
    }
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG( "IOCTL_LWLINK_OPTICAL_CHECK_EOM_STATUS  status=%d\n", pTrainParam->status );
    sendRequestCompletion();
    delete(pTrainParam);
}

// This function is lwrrently called for only internode links
void
LocalFMLWLinkReqConnTrainParallel::doParallelGetLinkStateReq( lwswitch::fmMessage *pFmMessage )
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelGetLinkStateReq " );

    lwlink_get_link_state *pTrainParam = new(lwlink_get_link_state);
    memset( pTrainParam, 0 , sizeof(lwlink_get_link_state) );

    FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel doParallelGetLinkStateReq size = %lu", mLwlinks.size() );
    pTrainParam->endPointCount = mLwlinks.size();
    // The info in mLwlinks[i].slaveEnd end is ignored
    for( unsigned int i = 0; i < pTrainParam->endPointCount; i++ ) {
        pTrainParam->endPoints[i].nodeId = mLwlinks[i].masterEnd.nodeId;
        pTrainParam->endPoints[i].linkIndex = mLwlinks[i].masterEnd.linkIndex;
        pTrainParam->endPoints[i].pciInfo = mLWLinkDevRepo->getDevicePCIInfo( mLwlinks[i].masterEnd.gpuOrSwitchId );

        FM_LOG_DEBUG( "LocalFMLWLinkReqConnTrainParallel: Parallel training get links state: (nodeId, bus, linkIndex)  %u, %u, %u " , 
                      pTrainParam->endPoints[i].nodeId, pTrainParam->endPoints[i].pciInfo.bus , pTrainParam->endPoints[i].linkIndex );

    }
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_GET_LINK_STATE, pTrainParam, sizeof(lwlink_get_link_state) );

    FM_LOG_DEBUG("LocalFMLWLinkReqConnTrainParallel doParallelGetLinkStateReq: returned endpoint count = %u", pTrainParam->endPointCount);
    for( unsigned int i = 0; i < pTrainParam->endPointCount; i++ ) {
        // only local end point is used so we always fill it in master
        mLwlinks[i].masterEnd.linkState.linkMode = pTrainParam->endState[i].linkMode;
        mLwlinks[i].masterEnd.linkState.txSubLinkMode = pTrainParam->endState[i].txSubLinkMode;
        mLwlinks[i].masterEnd.linkState.rxSubLinkMode = pTrainParam->endState[i].rxSubLinkMode;

        FM_LOG_DEBUG( "IOCTL (nodeId, linkIndex, busId) = (%u, %u, %u)", pTrainParam->endPoints[i].nodeId,
                      pTrainParam->endPoints[i].linkIndex, pTrainParam->endPoints[i].pciInfo.bus );
        FM_LOG_DEBUG( "IOCTL link state (%d,%d,%d)", pTrainParam->endState[i].linkMode,
                      pTrainParam->endState[i].txSubLinkMode, pTrainParam->endState[i].rxSubLinkMode );
    }
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(pTrainParam->status) );
    FM_LOG_DEBUG( "IOCTL_LWLINK_GET_LINK_STATE  status=%d\n", pTrainParam->status );
    sendRequestCompletion();
    delete(pTrainParam);

}

#endif
