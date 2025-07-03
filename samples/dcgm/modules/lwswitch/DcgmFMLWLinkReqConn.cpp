
#include <stdexcept>

#include "logging.h"
#include "DcgmFMLWLinkReqConn.h"
#include "DcgmFMLWLinkError.h"

DcgmFMLWLinkReqConn::DcgmFMLWLinkReqConn(lwswitch::fmMessage *pFmMessage,
                                         FMConnInterface *ctrlConnIntf,
                                         DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                         DcgmLFMLWLinkDevRepo *linkDevRepo)
    :DcgmFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConn::DcgmFMLWLinkReqConn\n" );

    lwswitch::lwlinkMsg linkMsg = pFmMessage->lwlinkmsg();
    lwswitch::lwlinkRequestMsg reqMsg = linkMsg.reqmsg();

    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            // copy the internode connection information from GFM
            parseAddInterNodeConnParams(reqMsg);
            break;
        }
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            // nothing specific to copy from GFM. LFM is returning conn info.
            break;
        }
        default:{
            PRINT_ERROR( "", "FMLWLinkReq: unknown request type in Add/Get Connection Handler" );
            throw std::runtime_error( "FMLWLinkReq: unknown request type in Add/Get Connection Handler" );
        }
    }
}

DcgmFMLWLinkReqConn::~DcgmFMLWLinkReqConn()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConn ~DcgmFMLWLinkReqConn\n" );
}

bool
DcgmFMLWLinkReqConn::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConn processNewMasterRequest\n" );
    // all the add/get connection requests are for local node and no co-ordination
    // with slave FM. So they complete from this context itself.
    bool bReqCompleted = true;    

    // get exclusive access to our request object
    lock();
    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            doAddInterNodeConn();
            break;
        }
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            doGetIntraNodeConns();
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown request type in Add/Get Connection\n" );
            break;
        }
    }

    // the request is finished. send response to global FM    
    sendRequestCompletion();
    unLock();
    return bReqCompleted;
}

bool
DcgmFMLWLinkReqConn::processReqTimeOut()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConn processReqTimeOut\n" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
DcgmFMLWLinkReqConn::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Add/Get Connection Req Information" << std::endl;

    // append base request dump information
    DcgmFMLWLinkReqBase::dumpInfo( os );
}

void
DcgmFMLWLinkReqConn::doAddInterNodeConn(void)
{
    int status = FM_LWLINK_ST_SUCCESS;

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_ADD_INTERNODE_CONN, &mInterNodeAddConnParam );
    if ( mInterNodeAddConnParam.status != LWL_SUCCESS ) {
        status = DcgmFMLWLinkError::getLinkErrorCode( mInterNodeAddConnParam.status );
    }

    setCompletionStatus( status );
}

void
DcgmFMLWLinkReqConn::doGetIntraNodeConns(void)
{
    DcgmFMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    DcgmFMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;
    uint32 connIdx = 0;

    // clean any cached intranode connections before starting new one
    mIntraNodeGetConnList.clear();

    // get intranode connection ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_get_intranode_conns getParam;
        memset(&getParam, 0, sizeof(getParam));
        DcgmFMLWLinkDevInfo lwlinkDevInfo = *it;
        getParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        getParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_DEVICE_GET_INTRANODE_CONNS, &getParam );
        if ( getParam.status != LWL_SUCCESS ) {
            status = DcgmFMLWLinkError::getLinkErrorCode( getParam.status );
            // clear any outstanding connection information
            mIntraNodeGetConnList.clear();
            break;
        } else {
            // the request is complete, copy connection information
            for ( connIdx = 0; connIdx < getParam.numConnections; connIdx++ ) {
                DcgmLWLinkConnInfo connInfo;
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
DcgmFMLWLinkReqConn::sendRequestCompletion(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqConn sendRequestCompletion\n" );

    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();

    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ADD_INTERNODE_CONN: {
            // copy our add connection results to google protobuf
            genAddInterNodeConnResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_GET_INTRANODE_CONNS: {
            // copy our get connection results to google protobuf
            genGetIntraNodeConnResp( rspMsg );
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown request type in Add/Get Connection\n" );
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
    // So use the actual dcgm requestId used by GFM as it is tracked.
    pFmMessage->set_requestid( getDcgmMsgReqId() );

    // Send final response to Global FM
    retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );

    if ( retVal != DCGM_ST_OK ) {
        // can't do much, just log an error
        PRINT_WARNING("", "error while sending train complete message to Fabric Manager\n");
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
DcgmFMLWLinkReqConn::genAddInterNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkAddInterNodeConnRspMsg *addConnRspMsg = new lwswitch::lwlinkAddInterNodeConnRspMsg();
    // no request specific data to return other than the overall status,
    // which is part of the lwlinkResponseMsg.
    rspMsg->set_allocated_addinternodeconnrspmsg( addConnRspMsg );
}

void
DcgmFMLWLinkReqConn::genGetIntraNodeConnResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkGetIntraNodeConnRspMsg *getConnRspMsg = new lwswitch::lwlinkGetIntraNodeConnRspMsg();
    DcgmLWLinkConnList::iterator it;

    // copy the received connection information from LWLink Driver 
    // as the response message to global FM.
    for ( it = mIntraNodeGetConnList.begin(); it != mIntraNodeGetConnList.end(); it++ ) {
        DcgmLWLinkConnInfo connInfo = *it;
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

void
DcgmFMLWLinkReqConn::parseAddInterNodeConnParams(lwswitch::lwlinkRequestMsg &reqMsg)
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

bool
DcgmFMLWLinkReqConn::isDuplicateConnection(DcgmLWLinkConnInfo &conn)
{
    DcgmLWLinkConnList::iterator it;
    for ( it = mIntraNodeGetConnList.begin(); it != mIntraNodeGetConnList.end(); it++ ) {
        DcgmLWLinkConnInfo tempConn = *it;
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
