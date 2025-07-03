
#include "logging.h"
#include <sstream>
#include <string.h>

#include "DcgmFMError.h"
#include "DcgmFMDevInfoMsgHndlr.h"
#include "DcgmLocalFabricManager.h"
#include "DcgmFMAutoLock.h"
#include "DcgmFMLWLinkError.h"

DcgmFMDevInfoMsgHdlrBase::DcgmFMDevInfoMsgHdlrBase(FMConnInterface *ctrlConnIntf)
{
    mCtrlConnIntf = ctrlConnIntf;
}

DcgmFMDevInfoMsgHdlrBase::~DcgmFMDevInfoMsgHdlrBase()
{
    // nothing specific to do
}

void
DcgmFMDevInfoMsgHdlrBase::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMDevInfoMsgHdlrBase handleMessage\n" );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ: {
            onLWLinkDevInfoReq( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_RSP: {
            onLWLinkDevInfoRsp( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ: {
            onLWSwitchDevInfoReq( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_RSP: {
            onLWSwitchDevInfoRsp( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ: {
            onGpuDevInfoReq( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_RSP: {
            onGpuDevInfoRsp( pFmMessage );
            break;
        }
        default: {
            PRINT_WARNING( "", "DcgmFMDevInfoMsgHdlrBase received unknown message type\n" );
            break;
        }
    }
}

LFMDevInfoMsgHdlr::LFMDevInfoMsgHdlr(DcgmLocalFabricManagerControl *pLfm)
    :DcgmFMDevInfoMsgHdlrBase( (FMConnInterface*)pLfm ),
     mpLfm( pLfm )
{
    // nothing specific to do
}

LFMDevInfoMsgHdlr::~LFMDevInfoMsgHdlr()
{
    // nothing specific to do
}

void
LFMDevInfoMsgHdlr::onLWLinkDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    // first get all the LWLink device information
    DcgmFMLWLinkDevInfoList lwlinkDevList;
    DcgmFMLWLinkDevInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkDeviceInfoRsp *linkInfoRsp = new lwswitch::lwlinkDeviceInfoRsp();

    // get all the locally present lwlink device information
    mpLfm->getAllLWLinkDevInfo( lwlinkDevList );

    for ( it = lwlinkDevList.begin(); it != lwlinkDevList.end(); it++ ) {
        DcgmFMLWLinkDevInfo lwlinkDevInfo = *it;
        lwlink_pci_dev_info pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        lwswitch::lwlinkDeviceInfoMsg *devInfoMsg = linkInfoRsp->add_devinfo();
        lwswitch::devicePciInfo *pciInfoMsg = new lwswitch::devicePciInfo();
        pciInfoMsg->set_domain( pciInfo.domain );
        pciInfoMsg->set_bus( pciInfo.bus );
        pciInfoMsg->set_device( pciInfo.device );
        pciInfoMsg->set_function( pciInfo.function);
        devInfoMsg->set_allocated_pciinfo( pciInfoMsg );
        devInfoMsg->set_devicename( lwlinkDevInfo.getDeviceName() );
        devInfoMsg->set_deviceid( lwlinkDevInfo.getDeviceId() );
        devInfoMsg->set_numlinks( lwlinkDevInfo.getNumLinks() );
        devInfoMsg->set_devtype( lwlinkDevInfo.getDeviceType() );
        devInfoMsg->set_enabledlinkmask( lwlinkDevInfo.getEnabledLinkMask() );
        devInfoMsg->set_uuid( (const char*)lwlinkDevInfo.getDeviceUuid() );
    }

    // create the final response message
    lwswitch::deviceInfoResponseMsg *rspMsg = new lwswitch::deviceInfoResponseMsg();
    rspMsg->set_inforeqid( reqMsg.inforeqid() );
    rspMsg->set_allocated_lwlinkdevrsp( linkInfoRsp );

    // fill the fabric message
    pFmRspMessage->set_type( lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_RSP );
    pFmRspMessage->set_allocated_devinforsp( rspMsg );
    pFmRspMessage->set_requestid( pFmMessage->requestid() );

    // send the message to GFM
    mCtrlConnIntf->SendMessageToGfm( pFmRspMessage, false );

    // free the allocated message and return final status
    delete( pFmRspMessage );
}

void
LFMDevInfoMsgHdlr::onLWSwitchDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    DcgmFMLWSwitchInfoList switchInfoList;
    DcgmFMLWSwitchInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::lwswitchDeviceInfoRsp *switchInfoRsp = new lwswitch::lwswitchDeviceInfoRsp();

    // get all the locally detected switch information
    mpLfm->getAllLwswitchInfo( switchInfoList );

    for (it = switchInfoList.begin() ; it != switchInfoList.end(); ++it) {
        DcgmFMLWSwitchInfo switchInfo = *it;
        lwswitch::lwswitchDeviceInfoMsg *devInfoMsg = switchInfoRsp->add_switchinfo();
        devInfoMsg->set_switchindex( switchInfo.switchIndex );
        devInfoMsg->set_physicalid( switchInfo.physicalId );
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        pciInfo->set_domain( switchInfo.pciInfo.domain );
        pciInfo->set_bus( switchInfo.pciInfo.bus );
        pciInfo->set_device( switchInfo.pciInfo.device );
        pciInfo->set_function( switchInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        devInfoMsg->set_enabledlinkmask( switchInfo.enabledLinkMask );
    }

    // create the final response message
    lwswitch::deviceInfoResponseMsg *rspMsg = new lwswitch::deviceInfoResponseMsg();
    rspMsg->set_inforeqid( reqMsg.inforeqid() );
    rspMsg->set_allocated_switchdevrsp( switchInfoRsp );

    // fill the fabric message
    pFmRspMessage->set_type( lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_RSP );
    pFmRspMessage->set_allocated_devinforsp( rspMsg );
    pFmRspMessage->set_requestid( pFmMessage->requestid() );

    // send the message to GFM
    mCtrlConnIntf->SendMessageToGfm( pFmRspMessage, false );

    // free the allocated message and return final status
    delete( pFmRspMessage );
}

void
LFMDevInfoMsgHdlr::onGpuDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    DcgmFMGpuInfoList gpuInfoList;
    DcgmFMGpuInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::gpuDeviceInfoRsp *gpuInfoRsp = new lwswitch::gpuDeviceInfoRsp();

    // get all the locally detected gpu information
    mpLfm->getAllGpuInfo( gpuInfoList);

    for (it = gpuInfoList.begin() ; it != gpuInfoList.end(); ++it) {
        DcgmFMGpuInfo gpuInfo = *it;
        lwswitch::gpuDeviceInfoMsg *devInfoMsg = gpuInfoRsp->add_gpuinfo();
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        pciInfo->set_domain( gpuInfo.pciInfo.domain );
        pciInfo->set_bus( gpuInfo.pciInfo.bus );
        pciInfo->set_device( gpuInfo.pciInfo.device );
        pciInfo->set_function( gpuInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        devInfoMsg->set_gpuindex( gpuInfo.gpuIndex );
        devInfoMsg->set_uuid( gpuInfo.uuid );
    }

    // get all the locally blacklisted gpu information
    DcgmFMGpuInfoList blacklistGpuInfoList;
    mpLfm->getBlacklistGpuInfo( blacklistGpuInfoList );
    for (it = blacklistGpuInfoList.begin() ; it != blacklistGpuInfoList.end(); ++it) {
        DcgmFMGpuInfo gpuInfo = *it;
        lwswitch::gpuDeviceInfoMsg *devInfoMsg = gpuInfoRsp->add_blacklistgpuinfo();
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        pciInfo->set_domain( gpuInfo.pciInfo.domain );
        pciInfo->set_bus( gpuInfo.pciInfo.bus );
        pciInfo->set_device( gpuInfo.pciInfo.device );
        pciInfo->set_function( gpuInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        devInfoMsg->set_gpuindex( gpuInfo.gpuIndex );
        devInfoMsg->set_uuid( gpuInfo.uuid );
    }

    // create the final response message
    lwswitch::deviceInfoResponseMsg *rspMsg = new lwswitch::deviceInfoResponseMsg();
    rspMsg->set_inforeqid( reqMsg.inforeqid() );
    rspMsg->set_allocated_gpudevrsp( gpuInfoRsp );

    // fill the fabric message
    pFmRspMessage->set_type( lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_RSP );
    pFmRspMessage->set_allocated_devinforsp( rspMsg );
    pFmRspMessage->set_requestid( pFmMessage->requestid() );

    // send the message to GFM
    mCtrlConnIntf->SendMessageToGfm( pFmRspMessage, false );

    // free the allocated message and return final status
    delete( pFmRspMessage );
}

void
LFMDevInfoMsgHdlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    // nothing specific for LFM to do in case of node disconnection
}

GFMDevInfoMsgHdlr::GFMDevInfoMsgHdlr(FMConnInterface *ctrlConnIntf)
    :DcgmFMDevInfoMsgHdlrBase( ctrlConnIntf )
{
    mDevInfoReqPending.clear();
    mDevInfoReqComplete.clear();
    lwosInitializeCriticalSection( &mLock );
    mNextInfoReqId = 0; //valid IDs start from 1
}

GFMDevInfoMsgHdlr::~GFMDevInfoMsgHdlr()
{
    mDevInfoReqPending.clear();
    mDevInfoReqComplete.clear();
    lwosDeleteCriticalSection( &mLock );    
}

dcgmReturn_t
GFMDevInfoMsgHdlr::sendLWLinkDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ, nodeId, requestId);
}

dcgmReturn_t
GFMDevInfoMsgHdlr::sendLWSwitchDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ, nodeId, requestId);
}

dcgmReturn_t
GFMDevInfoMsgHdlr::sendGpuDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ, nodeId, requestId);
}

dcgmReturn_t
GFMDevInfoMsgHdlr::sendDevInfoRequest(lwswitch::FabricManagerMessageType msgType,
                                      uint32 nodeId, uint64 &requestId)
{
    dcgmReturn_t retVal;
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    lwswitch::deviceInfoRequestMsg *reqMsg = new lwswitch::deviceInfoRequestMsg();
    uint64 infoReqId = 0;

    // fill the deviceInfoRequestMsg based on the request type
    switch( msgType ) {
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ: {
            lwswitch::lwlinkDeviceInfoReq *lwlinkReq = new lwswitch::lwlinkDeviceInfoReq();
            reqMsg->set_allocated_lwlinkdevreq(lwlinkReq);
            break;
        }
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ: {
            lwswitch::lwswitchDeviceInfoReq *switchReq = new lwswitch::lwswitchDeviceInfoReq();
            reqMsg->set_allocated_switchdevreq(switchReq);
            break;
        }
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ: {
            lwswitch::gpuDeviceInfoReq *gpuReq = new lwswitch::gpuDeviceInfoReq();
            reqMsg->set_allocated_gpudevreq(gpuReq);
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Link request type while parsing response message\n" );
            break;
        }
    }

    // fill the fabric message
    pFmMessage->set_type( msgType );

    // create the final train request message
    infoReqId = getNextDevInfoReqId();
    reqMsg->set_inforeqid(infoReqId);
    pFmMessage->set_allocated_devinforeq( reqMsg );

    // add request to our context for tracking
    // before sending the message, add it to the list as the response can 
    // come before even we add it to the list.
    addToReqPendingTable( infoReqId, nodeId );

    // send the message to the local FM node
    retVal = mCtrlConnIntf->SendMessageToLfm( nodeId, pFmMessage, true );

    if ( retVal != DCGM_ST_OK ) {
        // failed to send, remove the request from our local tracking
        removeFromReqPendingTable( infoReqId );
    } else {
        // send the request successfully.
        // update request id for caller to track
        requestId = infoReqId;
    }

    // free the allocated  message and return final status
    delete( pFmMessage );
    return retVal;
}


bool
GFMDevInfoMsgHdlr::isDevInfoReqComplete(uint64 &requestId, DevInfoReqResult &reqResult)
{
    DcgmFMAutoLock lock(mLock);
    DevInfoRequestMap::iterator it;

    it = mDevInfoReqComplete.find( requestId );
    if ( it != mDevInfoReqComplete.end() ) {
        DcgmDevInfoReqCtx reqCtx = it->second;
        reqResult = reqCtx.result;
        mDevInfoReqComplete.erase( requestId );
        return true;
    }

    return false;
}

void
GFMDevInfoMsgHdlr::onLWLinkDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // an LWLink device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GFMDevInfoMsgHdlr::onLWSwitchDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // an LWSwitch device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GFMDevInfoMsgHdlr::onGpuDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // GPU device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GFMDevInfoMsgHdlr::handleDevInfoReqCompletion(lwswitch::fmMessage *pFmMessage)
{
    lwswitch::deviceInfoResponseMsg devInfoRsp = pFmMessage->devinforsp();
    uint64 infoReqId = devInfoRsp.inforeqid();

    // find the request locally and update its result
    // also move the request to completed list from pending
    DcgmDevInfoReqCtx reqCtx = {0};
    if ( !getPendingInfoReq(infoReqId, reqCtx) ) {
        PRINT_WARNING( "%llu", "No device info request with ID %llu found during resp handling\n", infoReqId );
        return;
    }

    // update our local context with result information
    reqCtx.result.infoReqId = infoReqId;
    reqCtx.result.status = 0;
    reqCtx.result.devInfoRspMsg.CopyFrom(*pFmMessage);

    // remove from outstanding request and add to completed req table
    markPendingReqAsComplete(infoReqId, reqCtx);
}

void
GFMDevInfoMsgHdlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    PRINT_DEBUG( "", "GFMDevInfoMsgHdlr handleEvent\n" );

    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // do nothing
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            handleNodeDisconnect( nodeId );
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // dev info requests are initiated from GFM. So not applicable for GFM itself
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // dev info requests are initiated from GFM. So not applicable for GFM itself
            break;
        }
    }
}

void
GFMDevInfoMsgHdlr::addToReqPendingTable(uint64 reqId, uint32 toNodeId)
{
    DcgmFMAutoLock lock(mLock);

    // keep the request information locally and track
    DcgmDevInfoReqCtx reqCtx;
    memset( &reqCtx, 0, sizeof(reqCtx) );

    reqCtx.toNodeId = toNodeId;

    if ( mDevInfoReqPending.count(reqId) ) {
        PRINT_WARNING( "%llu", "Info request with Request ID %llu already present\n", reqId );
        return;
    }

    mDevInfoReqPending.insert( std::make_pair(reqId, reqCtx) );
}

void
GFMDevInfoMsgHdlr::removeFromReqPendingTable(uint64 reqId)
{
    DcgmFMAutoLock lock(mLock);

    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it != mDevInfoReqPending.end() ) {
        mDevInfoReqPending.erase( it );
    }
}

bool
GFMDevInfoMsgHdlr::getPendingInfoReq(uint64 reqId, DcgmDevInfoReqCtx &reqCtx)
{
    DcgmFMAutoLock lock(mLock);

    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it == mDevInfoReqPending.end() ) {
        return false;
    }

    // found the request, copy it and return success
    reqCtx = it->second;
    return true;
}

void
GFMDevInfoMsgHdlr::markPendingReqAsComplete(uint64 reqId, DcgmDevInfoReqCtx &reqCtx)
{
    DcgmFMAutoLock lock(mLock);
    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it == mDevInfoReqPending.end() ) {
        return;
    }

    // erase from pending and add to complete list
    mDevInfoReqPending.erase( it );
    mDevInfoReqComplete.insert( std::make_pair(reqId, reqCtx) );
}

uint64
GFMDevInfoMsgHdlr::getNextDevInfoReqId(void)
{
    PRINT_DEBUG( "", "GFMDevInfoMsgHdlr getNextDevInfoReqId\n" );
    DcgmFMAutoLock lock(mLock);

    mNextInfoReqId++;
    if ( mNextInfoReqId == 0 ) {
        // wrap around
        mNextInfoReqId = 1;
    }
    return mNextInfoReqId;
}

void
GFMDevInfoMsgHdlr::handleNodeDisconnect(uint32 nodeId)
{
    // node's socket connection got disconnected. complete all the
    // outstanding request to this LFM node as completed with
    // appropriate error code.
    DcgmFMAutoLock lock(mLock);

    PRINT_DEBUG( "", "GFMDevInfoMsgHdlr handleNodeDisconnect\n" );

    DevInfoRequestMap::iterator it = mDevInfoReqPending.begin();
    while ( it != mDevInfoReqPending.end() ) {
        DcgmDevInfoReqCtx reqCtx = it->second;
        if ( reqCtx.toNodeId == nodeId ) {
            // move to the completed list as failure
            uint64 reqId = it->first;
            reqCtx.result.infoReqId = reqId;
            reqCtx.result.status = FM_LWLINK_ST_MASTER_FM_SOCKET_ERR;
            mDevInfoReqComplete.insert( std::make_pair(reqId, reqCtx) );
            mDevInfoReqPending.erase( it++ );
            continue;
        }
        ++it;
    }
}
