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
#include "LocalFMLWLinkReqConn.h"
#include "FMLWLinkError.h"

LocalFMLWLinkReqConn::LocalFMLWLinkReqConn(lwswitch::fmMessage *pFmMessage,
                                           FMConnInterface *ctrlConnIntf,
                                           LocalFMLWLinkDrvIntf *linkDrvIntf,
                                           LocalFMLWLinkDevRepo *linkDevRepo)
    :LocalFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConn::LocalFMLWLinkReqConn" );

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkRequestMsg reqMsg = linkMsg.reqmsg();

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    mInterNodeAddConnParam = {{0}};
#endif

    switch ( getReqType() ) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            // copy the internode connection information from GFM
            parseAddInterNodeConnParams(reqMsg);
            break;
        }
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            // nothing specific to copy from GFM. LFM is returning conn info.
            break;
        }
        default:{
            std::ostringstream ss;
            ss << "lwlink request conn: unknown LWLink request type detected in connection handler";
            FM_LOG_ERROR("%s", ss.str().c_str());
        }
    }
}

LocalFMLWLinkReqConn::~LocalFMLWLinkReqConn()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConn ~LocalFMLWLinkReqConn" );
}

bool
LocalFMLWLinkReqConn::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConn processNewMasterRequest" );
    // all the add/get connection requests are for local node and no co-ordination
    // with slave FM. So they complete from this context itself.
    bool bReqCompleted = true;    

    // get exclusive access to our request object
    lock();
    switch ( getReqType() ) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            doAddInterNodeConn();
            break;
        }
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            doGetIntraNodeConns();
            break;
        }
        default:{
            FM_LOG_ERROR( "Unknown request type in link training add/get connection handler" );
            break;
        }
    }

    // the request is finished. send response to global FM    
    sendRequestCompletion();
    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqConn::processReqTimeOut()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConn processReqTimeOut" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
LocalFMLWLinkReqConn::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Add/Get Connection Req Information" << std::endl;

    // append base request dump information
    LocalFMLWLinkReqBase::dumpInfo( os );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConn::doAddInterNodeConn(void)
{
    int status = FM_LWLINK_ST_SUCCESS;

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_ADD_INTERNODE_CONN, 
                             &mInterNodeAddConnParam, sizeof(mInterNodeAddConnParam) );
    if ( mInterNodeAddConnParam.status != LWL_SUCCESS ) {
        status = FMLWLinkError::getLinkErrorCode( mInterNodeAddConnParam.status );
    }

    setCompletionStatus( status );
}
#endif

void
LocalFMLWLinkReqConn::doGetIntraNodeConns(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;
    uint32 connIdx = 0;

    // clean any cached intranode connections before starting new one
    mIntraNodeGetConnList.clear();

    // get intranode connection ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_get_intranode_conns getParam;
        memset(&getParam, 0, sizeof(getParam));
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        getParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        getParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &getParam, sizeof(getParam) );
        if ( getParam.status != LWL_SUCCESS ) {
            status = FMLWLinkError::getLinkErrorCode( getParam.status );
            // clear any outstanding connection information
            mIntraNodeGetConnList.clear();
            break;
        } else {
            // the request is complete, copy connection information
            for ( connIdx = 0; connIdx < getParam.numConnections; connIdx++ ) {
                FMLWLinkConnInfo connInfo;
                lwlink_pci_dev_info pciInfo;
                // copy source device information
                connInfo.masterEnd.nodeId = getParam.conn[connIdx].srcEndPoint.nodeId;
                connInfo.masterEnd.linkIndex = getParam.conn[connIdx].srcEndPoint.linkIndex;
                pciInfo = getParam.conn[connIdx].srcEndPoint.pciInfo;
                connInfo.masterEnd.gpuOrSwitchId = mLWLinkDevRepo->getDeviceId(pciInfo);
                // copy destination device information                
                connInfo.slaveEnd.nodeId = getParam.conn[connIdx].dstEndPoint.nodeId;
                connInfo.slaveEnd.linkIndex = getParam.conn[connIdx].dstEndPoint.linkIndex;
                pciInfo = getParam.conn[connIdx].dstEndPoint.pciInfo;
                connInfo.slaveEnd.gpuOrSwitchId = mLWLinkDevRepo->getDeviceId(pciInfo);
                // check whether we have already got this connection. Since the get connection
                // ioctl is per device, we can get the same connection pair which belongs to
                // two different LWLink device
                if ( !isDuplicateConnection(connInfo) ) {
                    mIntraNodeGetConnList.push_back(connInfo);
                }
            }
        }
    }

    setCompletionStatus( status );
}

void
LocalFMLWLinkReqConn::sendRequestCompletion(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqConn sendRequestCompletion" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();

    switch ( getReqType() ) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            // copy our add connection results to google protobuf
            genAddInterNodeConnResp( rspMsg );
            break;
        }
#endif
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            // copy our get connection results to google protobuf
            genGetIntraNodeConnResp( rspMsg );
            break;
        }
        default:{
            FM_LOG_ERROR( "request type is invalid in lwlink connection handler when sending request completion" );
            break;
        }
    }

    rspMsg->set_status( getCompletionStatus() );

    // create the final train request message
    lwswitch::lwlinkMsg* linkMsg = new lwswitch::lwlinkMsg();
    linkMsg->set_trainreqid( getTrainReqId() );
    linkMsg->set_allocated_rspmsg( rspMsg );

    // fill the fabric message
    pFmMessage->set_type( lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE );
    pFmMessage->set_allocated_lwlinkmsg( linkMsg );
    // the request is completed and the response is to GFM.
    // So use the actual FM msg requestId used by GFM as it is tracked.
    pFmMessage->set_requestid( getFmMsgHdrReqId() );

    // Send final response to Global FM
    retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // can't do much, just log an error
        FM_LOG_WARNING( "error while sending train complete message to fabric manager" );
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConn::genAddInterNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkAddInterNodeConnRspMsg *addConnRspMsg = new lwswitch::lwlinkAddInterNodeConnRspMsg();
    // no request specific data to return other than the overall status,
    // which is part of the lwlinkResponseMsg.
    rspMsg->set_allocated_addinternodeconnrspmsg( addConnRspMsg );
}
#endif

void
LocalFMLWLinkReqConn::genGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkGetIntraNodeConnRspMsg *getConnRspMsg = new lwswitch::lwlinkGetIntraNodeConnRspMsg();
    FMLWLinkConnList::iterator it;

    // copy the received connection information from LWLink Driver 
    // as the response message to global FM.
    for ( it = mIntraNodeGetConnList.begin(); it != mIntraNodeGetConnList.end(); it++ ) {
        FMLWLinkConnInfo connInfo = *it;
        lwswitch::lwlinkConnectionInfo *connInfoMsg = getConnRspMsg->add_conninfo();
        lwswitch::lwlinkEndPointInfo *masterEnd = new lwswitch::lwlinkEndPointInfo();
        lwswitch::lwlinkEndPointInfo *slaveEnd = new lwswitch::lwlinkEndPointInfo();

         // copy source device information
        masterEnd->set_nodeid( connInfo.masterEnd.nodeId );
        masterEnd->set_gpuorswitchid( connInfo.masterEnd.gpuOrSwitchId );
        masterEnd->set_linkindex( connInfo.masterEnd.linkIndex);
         // copy destination device information
        slaveEnd->set_nodeid( connInfo.slaveEnd.nodeId );
        slaveEnd->set_gpuorswitchid( connInfo.slaveEnd.gpuOrSwitchId );
        slaveEnd->set_linkindex( connInfo.slaveEnd.linkIndex);
        connInfoMsg->set_allocated_masterend( masterEnd );
        connInfoMsg->set_allocated_slaveend( slaveEnd );
    }

    rspMsg->set_allocated_getintranodeconnrspmsg( getConnRspMsg );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqConn::parseAddInterNodeConnParams(lwswitch::lwlinkRequestMsg &reqMsg)
{
    memset( &mInterNodeAddConnParam, 0 , sizeof(mInterNodeAddConnParam) );
    lwswitch::lwlinkAddInterNodeConnReqMsg addConnMsg = reqMsg.addinternodeconnreqmsg();
    lwswitch::lwlinkInterNodeConnInfo connInfo = addConnMsg.conninfo();
    lwswitch::lwlinkEndPointInfo localEnd = connInfo.localend();
    lwswitch::lwlinkRemoteEndPointInfo remoteEnd = connInfo.remoteend();
    lwswitch::devicePciInfo remotePciInfo = remoteEnd.pciinfo();

    // copy the add connection information from GFM
    mInterNodeAddConnParam.localEndPoint.nodeId = localEnd.nodeid();
    mInterNodeAddConnParam.localEndPoint.linkIndex = localEnd.linkindex();
    mInterNodeAddConnParam.localEndPoint.pciInfo = mLWLinkDevRepo->getDevicePCIInfo( localEnd.gpuorswitchid() );

    mInterNodeAddConnParam.remoteEndPoint.nodeId = remoteEnd.nodeid();
    mInterNodeAddConnParam.remoteEndPoint.linkIndex = remoteEnd.linkindex();
    lwlink_pci_dev_info pciInfo;
    pciInfo.domain = remotePciInfo.domain();
    pciInfo.bus = remotePciInfo.bus();
    pciInfo.device = remotePciInfo.device();
    pciInfo.function = remotePciInfo.function();
    mInterNodeAddConnParam.remoteEndPoint.pciInfo = pciInfo;
    mInterNodeAddConnParam.remoteEndPoint.devType = remoteEnd.devtype();
    if ( remoteEnd.has_uuid() ) {
        memcpy( mInterNodeAddConnParam.remoteEndPoint.devUuid, remoteEnd.uuid().c_str(),
                sizeof(mInterNodeAddConnParam.remoteEndPoint.devUuid) );
    }
}
#endif

bool
LocalFMLWLinkReqConn::isDuplicateConnection(FMLWLinkConnInfo &conn)
{
    FMLWLinkConnList::iterator it;
    for ( it = mIntraNodeGetConnList.begin(); it != mIntraNodeGetConnList.end(); it++ ) {
        FMLWLinkConnInfo tempConn = *it;
        if ( (tempConn.masterEnd == conn.masterEnd) &&
             (tempConn.slaveEnd== conn.slaveEnd) ) {
            return true;
        }
        // do cross-match as well
        if ( (tempConn.masterEnd == conn.slaveEnd) &&
             (tempConn.slaveEnd== conn.masterEnd) ) {
            return true;
        }
    }

    return false;
}
