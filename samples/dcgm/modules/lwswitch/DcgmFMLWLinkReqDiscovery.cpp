
#include "logging.h"
#include "DcgmFMLWLinkReqDiscovery.h"
#include "DcgmFMLWLinkError.h"

DcgmFMLWLinkReqDiscovery::DcgmFMLWLinkReqDiscovery(lwswitch::fmMessage *pFmMessage,
                                                   FMConnInterface *ctrlConnIntf,
                                                   DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                                   DcgmLFMLWLinkDevRepo *linkDevRepo)
    :DcgmFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqDiscovery::DcgmFMLWLinkReqDiscovery\n" );

    mWriteTokenList.clear();
    mReadTokenList.clear();
}

DcgmFMLWLinkReqDiscovery::~DcgmFMLWLinkReqDiscovery()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqDiscovery ~DcgmFMLWLinkReqDiscovery\n" );
    mWriteTokenList.clear();
    mReadTokenList.clear();
}

bool
DcgmFMLWLinkReqDiscovery::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqDiscovery processNewMasterRequest\n" );

    // all the discovery requests are for local node and no co-ordination with
    // slave FM. So they complete from this context itself.
    bool bReqCompleted = true;    

    // get exclusive access to our request object
    lock();

    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS: {
            doDiscoverIntraNodeConns();
            break;
        }
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS: {
            doWriteDiscoveryToken();
            break;
        }
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS: {
            doReadDiscoveryToken();
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Link Discover request type\n" );
            break;
        }
    }

    // the request is finished. send response to global FM    
    sendRequestCompletion();
    unLock();
    return bReqCompleted;
}

bool
DcgmFMLWLinkReqDiscovery::processReqTimeOut()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqDiscovery processReqTimeOut\n" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
DcgmFMLWLinkReqDiscovery::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Node Discovery/Connection Req Information" << std::endl;

    // append base request dump information
    DcgmFMLWLinkReqBase::dumpInfo( os );
}

void
DcgmFMLWLinkReqDiscovery::doDiscoverIntraNodeConns(void)
{
    lwlink_discover_intranode_conns discoverParam = {0};

    // discover connection is for the whole node and it shouldn't take much time 
    // as it simply writing/reading tokens, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discoverParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(discoverParam.status) );
}

void
DcgmFMLWLinkReqDiscovery::doWriteDiscoveryToken(void)
{
    DcgmFMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    DcgmFMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;
    uint32 tokenIdx = 0;

    /*
     * TODO/NOTE: Write discovery token ioctl is per device. If required, we can
     * make per device thread to issue the ioctl to improve the overall exelwtion time
     */

    // clean any cached token list before starting new one
    mWriteTokenList.clear();

    // write token ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_write_discovery_tokens writeParam;
        memset( &writeParam, 0, sizeof(writeParam));
        DcgmFMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        writeParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        writeParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();

        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS, &writeParam );
        if ( writeParam.status != LWL_SUCCESS ) {
            status = DcgmFMLWLinkError::getLinkErrorCode( writeParam.status );
            // clear any outstanding tokens as the request is marked as failed
            mWriteTokenList.clear();
            break;
        } else {
            // this request is completed successfully, copy the token information
            // locally, which should be send to GFM as part of the reply.
            for (tokenIdx = 0; tokenIdx < writeParam.numTokens; tokenIdx++) {
                DcgmLinkDiscoveryTokenInfo tokenInfo;
                tokenInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
                tokenInfo.gpuOrSwitchId = lwlinkDevInfo.getDeviceId();
                tokenInfo.linkIndex = writeParam.tokenInfo[tokenIdx].linkIndex;
                tokenInfo.tokelwalue = writeParam.tokenInfo[tokenIdx].tokelwalue;
                mWriteTokenList.push_back(tokenInfo);
            }
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
DcgmFMLWLinkReqDiscovery::doReadDiscoveryToken(void)
{
    DcgmFMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    DcgmFMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;
    uint32 tokenIdx = 0;

    /*
     * TODO/NOTE: Read discovery token ioctl is per device. If required, we can
     * make per device thread to issue the ioctl to improve the overall exelwtion time
     */

    // clean any cached token list before starting new one
    mReadTokenList.clear();

    // read token ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_read_discovery_tokens readParam;
        memset( &readParam, 0, sizeof(readParam));
        DcgmFMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        readParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        readParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_READ_DISCOVERY_TOKENS, &readParam );
        if ( readParam.status != LWL_SUCCESS ) {
            status = DcgmFMLWLinkError::getLinkErrorCode( readParam.status );
            // clear any outstanding tokens as the request is marked as failed
            mReadTokenList.clear();
            break;
        } else {
            // this request is completed successfully, copy the token information
            // locally, which should be send to GFM as part of the reply.
            for (tokenIdx = 0; tokenIdx < readParam.numTokens; tokenIdx++) {
                DcgmLinkDiscoveryTokenInfo tokenInfo;
                tokenInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
                tokenInfo.gpuOrSwitchId = lwlinkDevInfo.getDeviceId();
                tokenInfo.linkIndex = readParam.tokenInfo[tokenIdx].linkIndex;
                tokenInfo.tokelwalue = readParam.tokenInfo[tokenIdx].tokelwalue;
                mReadTokenList.push_back(tokenInfo);
            }
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
DcgmFMLWLinkReqDiscovery::sendRequestCompletion(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqDiscovery sendRequestCompletion\n" );

    // send the response to GFM
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();

    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_DISCOVER_INTRANODE_CONNS: {
            genDiscoverIntraNodeConnsResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_WRITE_DISCOVERY_TOKENS: {
            genWriteDiscoveryTokenResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_READ_DISCOVERY_TOKENS: {
            genReadDiscoveryTokenResp( rspMsg );
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Link Discover request type while preparing response\n" );
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

    // Send final response to GFM
    retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );

    if ( retVal != DCGM_ST_OK ) {
        // can't do much, just log an error
        PRINT_WARNING("", "error while sending train complete message to Fabric Manager\n");
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
DcgmFMLWLinkReqDiscovery::genDiscoverIntraNodeConnsResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkDiscoverIntraNodeConnRspMsg* discConnRspMsg = new lwswitch::lwlinkDiscoverIntraNodeConnRspMsg();
    // no request specific data to return other than the overall status,
    // which is part of the lwlinkResponseMsg.
    rspMsg->set_allocated_discoverintranodeconnrspmsg( discConnRspMsg );
}

void
DcgmFMLWLinkReqDiscovery::genWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkWriteDiscoveryTokenRspMsg *tokenRspMsg = new lwswitch::lwlinkWriteDiscoveryTokenRspMsg();
    DcgmLWLinkDiscoveryTokenList::iterator it;

    // copy the token information to global FM.
    for ( it = mWriteTokenList.begin(); it != mWriteTokenList.end(); it++ ) {
        DcgmLinkDiscoveryTokenInfo tokenInfo = *it;
        lwswitch::lwlinkDiscoveryTokenInfo *tokenInfoMsg = tokenRspMsg->add_tokeninfo();

         // copy token and device information
        tokenInfoMsg->set_nodeid( tokenInfo.nodeId );
        tokenInfoMsg->set_gpuorswitchid( tokenInfo.gpuOrSwitchId );
        tokenInfoMsg->set_linkindex( tokenInfo.linkIndex);
        tokenInfoMsg->set_tokelwalue( tokenInfo.tokelwalue );
    }

    rspMsg->set_allocated_writedisctokenrspmsg( tokenRspMsg );
}

void
DcgmFMLWLinkReqDiscovery::genReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkReadDiscoveryTokenRspMsg *tokenRspMsg = new lwswitch::lwlinkReadDiscoveryTokenRspMsg();
    DcgmLWLinkDiscoveryTokenList::iterator it;

    // copy the token information to global FM.
    for ( it = mReadTokenList.begin(); it != mReadTokenList.end(); it++ ) {
        DcgmLinkDiscoveryTokenInfo tokenInfo = *it;
        lwswitch::lwlinkDiscoveryTokenInfo *tokenInfoMsg = tokenRspMsg->add_tokeninfo();

         // copy token and device information
        tokenInfoMsg->set_nodeid( tokenInfo.nodeId );
        tokenInfoMsg->set_gpuorswitchid( tokenInfo.gpuOrSwitchId );
        tokenInfoMsg->set_linkindex( tokenInfo.linkIndex);
        tokenInfoMsg->set_tokelwalue( tokenInfo.tokelwalue );
    }

    rspMsg->set_allocated_readdisctokenrspmsg( tokenRspMsg );
}
