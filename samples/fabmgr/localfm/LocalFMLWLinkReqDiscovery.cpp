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
#include "LocalFMLWLinkReqDiscovery.h"
#include "FMLWLinkError.h"

LocalFMLWLinkReqDiscovery::LocalFMLWLinkReqDiscovery(lwswitch::fmMessage *pFmMessage,
                                                     FMConnInterface *ctrlConnIntf,
                                                     LocalFMLWLinkDrvIntf *linkDrvIntf,
                                                     LocalFMLWLinkDevRepo *linkDevRepo)
    :LocalFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqDiscovery::LocalFMLWLinkReqDiscovery" );

    mWriteTokenList.clear();
    mReadTokenList.clear();
}

LocalFMLWLinkReqDiscovery::~LocalFMLWLinkReqDiscovery()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqDiscovery ~LocalFMLWLinkReqDiscovery" );
    mWriteTokenList.clear();
    mReadTokenList.clear();
}

bool
LocalFMLWLinkReqDiscovery::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqDiscovery processNewMasterRequest" );

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
        case lwswitch::FM_LWLINK_READ_SIDS: {
            doReadSid();
            break;
        }
        default:{
            FM_LOG_ERROR( "unknown request type %d detected in LWLink discover connection handler",
                          getReqType() );
            break;
        }
    }

    // the request is finished. send response to global FM    
    sendRequestCompletion();
    unLock();
    return bReqCompleted;
}

bool
LocalFMLWLinkReqDiscovery::processReqTimeOut()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqDiscovery processReqTimeOut" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock();
    return bReqCompleted;
}

void
LocalFMLWLinkReqDiscovery::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Node Discovery/Connection Req Information" << std::endl;

    // append base request dump information
    LocalFMLWLinkReqBase::dumpInfo( os );
}

void
LocalFMLWLinkReqDiscovery::doDiscoverIntraNodeConns(void)
{
    lwlink_discover_intranode_conns discoverParam = {0};

    // discover connection is for the whole node and it shouldn't take much time 
    // as it simply writing/reading tokens, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_DISCOVER_INTRANODE_CONNS, &discoverParam, sizeof(discoverParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(discoverParam.status) );
}

void
LocalFMLWLinkReqDiscovery::doWriteDiscoveryToken(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
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
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        writeParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        writeParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();

        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_WRITE_DISCOVERY_TOKENS, &writeParam, sizeof(writeParam) );
        if ( writeParam.status != LWL_SUCCESS ) {
            status = FMLWLinkError::getLinkErrorCode( writeParam.status );
            // clear any outstanding tokens as the request is marked as failed
            mWriteTokenList.clear();
            break;
        } else {
            // this request is completed successfully, copy the token information
            // locally, which should be send to GFM as part of the reply.
            for (tokenIdx = 0; tokenIdx < writeParam.numTokens; tokenIdx++) {
                FMLinkDiscoveryTokenInfo tokenInfo;
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
LocalFMLWLinkReqDiscovery::doReadSid(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;
    uint32 linkIdx = 0;
    FM_LOG_DEBUG("LocalFMLWLinkReqDiscovery::doReadSid");

    /*
     * TODO/NOTE: Read discovery token ioctl is per device. If required, we can
     * make per device thread to issue the ioctl to improve the overall exelwtion time
     */

    // clean any cached token list before starting new one
    mReadSidList.clear();

    // read token ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_read_sids readParam;
        FM_LOG_DEBUG("LocalFMLWLinkReqDiscovery::doReadSid calling ioctl");

        memset( &readParam, 0, sizeof(readParam));

        FMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        readParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        readParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();

        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_DEVICE_READ_SIDS, &readParam, sizeof(readParam) );
        if ( readParam.status != LWL_SUCCESS ) {
            status = FMLWLinkError::getLinkErrorCode( readParam.status );
            // clear any outstanding tokens as the request is marked as failed
            mReadSidList.clear();
            break;
        } else {
            // this request is completed successfully, copy the token information
            // locally, which should be send to GFM as part of the reply.
            FM_LOG_DEBUG("LocalFMLWLinkReqDiscovery::doReadSid numEntries=%d", readParam.numEntries);
            for (linkIdx = 0; linkIdx < readParam.numEntries; linkIdx++) {
                FMLinkSidInfo sidInfo;

                sidInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
                sidInfo.gpuOrSwitchId = lwlinkDevInfo.getDeviceId();
                sidInfo.nearSid = readParam.sidInfo[linkIdx].localLinkSid;
                sidInfo.nearLinkIndex = readParam.sidInfo[linkIdx].localLinkNum;
                sidInfo.farSid = readParam.sidInfo[linkIdx].remoteLinkSid;
                sidInfo.farLinkIndex = readParam.sidInfo[linkIdx].remoteLinkNum;
                FM_LOG_DEBUG("nodeId=%d gpuOrSwitchId=%llu nearSid=%llu nearLinkIndex=%d", 
                             sidInfo.nodeId, sidInfo.gpuOrSwitchId, sidInfo.nearSid, sidInfo.nearLinkIndex);
                FM_LOG_DEBUG("farSid=%llu farLinkIndex=%d", sidInfo.farSid, sidInfo.farLinkIndex);
                // TODO : keeping this workaround for driver returning junk for bad GPU links as though they are for remote node
                //sidInfo.farSid = sidInfo.nodeId ? 0: 1;
                //sidInfo.nearSid = sidInfo.nodeId;
                // TODO: hard-coded for E4700 based 2-node system as links 0-12 connect to local node GPU.
                //if (sidInfo.nearLinkIndex < 12)
                  //  continue;
                mReadSidList.push_back(sidInfo);
            }
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqDiscovery::doReadDiscoveryToken(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
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
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        readParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        readParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
        mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_READ_DISCOVERY_TOKENS, &readParam, sizeof(readParam) );
        if ( readParam.status != LWL_SUCCESS ) {
            status = FMLWLinkError::getLinkErrorCode( readParam.status );
            // clear any outstanding tokens as the request is marked as failed
            mReadTokenList.clear();
            break;
        } else {
            // this request is completed successfully, copy the token information
            // locally, which should be send to GFM as part of the reply.
            for (tokenIdx = 0; tokenIdx < readParam.numTokens; tokenIdx++) {
                FMLinkDiscoveryTokenInfo tokenInfo;
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
LocalFMLWLinkReqDiscovery::sendRequestCompletion(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqDiscovery sendRequestCompletion" );

    // send the response to GFM
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;
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
        case lwswitch::FM_LWLINK_READ_SIDS: {
            genReadSidResp( rspMsg );
            break;
        }
        default:{
            FM_LOG_ERROR( "unknown LWLink discover connection request type while preparing response" );
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

    // Send final response to GFM
    retVal = mCtrlConnIntf->SendMessageToGfm( pFmMessage, false );

    if ( retVal != FM_INT_ST_OK ) {
        // can't do much, just log an error
        FM_LOG_WARNING( "error while sending LWLink discover connection complete message to fabric manager" );
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
}

void
LocalFMLWLinkReqDiscovery::genDiscoverIntraNodeConnsResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkDiscoverIntraNodeConnRspMsg* discConnRspMsg = new lwswitch::lwlinkDiscoverIntraNodeConnRspMsg();
    // no request specific data to return other than the overall status,
    // which is part of the lwlinkResponseMsg.
    rspMsg->set_allocated_discoverintranodeconnrspmsg( discConnRspMsg );
}

void
LocalFMLWLinkReqDiscovery::genWriteDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkWriteDiscoveryTokenRspMsg *tokenRspMsg = new lwswitch::lwlinkWriteDiscoveryTokenRspMsg();
    FMLWLinkDiscoveryTokenList::iterator it;

    // copy the token information to global FM.
    for ( it = mWriteTokenList.begin(); it != mWriteTokenList.end(); it++ ) {
        FMLinkDiscoveryTokenInfo tokenInfo = *it;
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
LocalFMLWLinkReqDiscovery::genReadSidResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkReadSidRspMsg *sidRspMsg = new lwswitch::lwlinkReadSidRspMsg();
    FMLWLinkSidList::iterator it;

    // copy the token information to global FM.
    for ( it = mReadSidList.begin(); it != mReadSidList.end(); it++ ) {
        FMLinkSidInfo sidInfo = *it;
        lwswitch::lwlinkSidInfo *sidInfoMsg = sidRspMsg->add_sidinfo();

         // copy token and device information
        sidInfoMsg->set_nodeid( sidInfo.nodeId );
        sidInfoMsg->set_gpuorswitchid( sidInfo.gpuOrSwitchId );
        sidInfoMsg->set_nearsid( sidInfo.nearSid);
        sidInfoMsg->set_nearlinkindex( sidInfo.nearLinkIndex);
        sidInfoMsg->set_farsid( sidInfo.farSid);
        sidInfoMsg->set_farlinkindex( sidInfo.farLinkIndex);
    }
    rspMsg->set_allocated_readsidrspmsg( sidRspMsg );
}
void
LocalFMLWLinkReqDiscovery::genReadDiscoveryTokenResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkReadDiscoveryTokenRspMsg *tokenRspMsg = new lwswitch::lwlinkReadDiscoveryTokenRspMsg();
    FMLWLinkDiscoveryTokenList::iterator it;

    // copy the token information to global FM.
    for ( it = mReadTokenList.begin(); it != mReadTokenList.end(); it++ ) {
        FMLinkDiscoveryTokenInfo tokenInfo = *it;
        lwswitch::lwlinkDiscoveryTokenInfo *tokenInfoMsg = tokenRspMsg->add_tokeninfo();

         // copy token and device information
        tokenInfoMsg->set_nodeid( tokenInfo.nodeId );
        tokenInfoMsg->set_gpuorswitchid( tokenInfo.gpuOrSwitchId );
        tokenInfoMsg->set_linkindex( tokenInfo.linkIndex);
        tokenInfoMsg->set_tokelwalue( tokenInfo.tokelwalue );
    }

    rspMsg->set_allocated_readdisctokenrspmsg( tokenRspMsg );
}
