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
#include <sstream>
#include <string.h>

#include "FMErrorCodesInternal.h"
#include "FMDevInfoMsgHndlr.h"
#include "LocalFabricManager.h"
#include "FMAutoLock.h"
#include "FMLWLinkError.h"
#include "FMVersion.h"

FMDevInfoMsgHdlrBase::FMDevInfoMsgHdlrBase(FMConnInterface *ctrlConnIntf)
{
    mCtrlConnIntf = ctrlConnIntf;
}

FMDevInfoMsgHdlrBase::~FMDevInfoMsgHdlrBase()
{
    // nothing specific to do
}

void
FMDevInfoMsgHdlrBase::handleMessage(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "FMDevInfoMsgHdlrBase handleMessage\n" );

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
        case lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_REQ: {
            onGpuLWLinkSpeedInfoReq( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_RSP: {
            onGpuLWLinkSpeedInfoRsp( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_VERSION_INFO_REQ: {
            onGetFMVersionInfoReq( pFmMessage );
            break;
        }
        case lwswitch::FM_NODE_GET_VERSION_INFO_RSP: {
            onGetFMVersionInfoRsp( pFmMessage );
            break;
        }
        default: {
            FM_LOG_WARNING( "device info message handler: received unknown message type: %d\n",
                            pFmMessage->type() );
            break;
        }
    }
}

LocalFMDevInfoMsgHdlr::LocalFMDevInfoMsgHdlr(LocalFabricManagerControl *pLfm)
    :FMDevInfoMsgHdlrBase( (FMConnInterface*)pLfm ),
     mpLfm( pLfm )
{
    // nothing specific to do
}

LocalFMDevInfoMsgHdlr::~LocalFMDevInfoMsgHdlr()
{
    // nothing specific to do
}

void
LocalFMDevInfoMsgHdlr::onLWLinkDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    // first get all the LWLink device information
    FMLWLinkDevInfoList lwlinkDevList;
    FMLWLinkDevInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::lwlinkDeviceInfoRsp *linkInfoRsp = new lwswitch::lwlinkDeviceInfoRsp();
    // get all the locally present lwlink device information
    lwlinkDevList.clear();
    mpLfm->getAllLWLinkDevInfo( lwlinkDevList );

    for ( it = lwlinkDevList.begin(); it != lwlinkDevList.end(); it++ ) {
        FMLWLinkDevInfo lwlinkDevInfo = *it;
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
#endif // LW_MODS_GDM_BUILD
}

void
LocalFMDevInfoMsgHdlr::onLWSwitchDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    FMLWSwitchInfoList switchInfoList;
    FMLWSwitchInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::lwswitchDeviceInfoRsp *switchInfoRsp = new lwswitch::lwswitchDeviceInfoRsp();

    // get all the locally detected switch information
    switchInfoList.clear();
    mpLfm->getAllLwswitchInfo( switchInfoList );

    for (it = switchInfoList.begin() ; it != switchInfoList.end(); ++it) {
        FMLWSwitchInfo switchInfo = *it;
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
        devInfoMsg->set_archtype( switchInfo.archType );
        devInfoMsg->set_uuid( switchInfo.uuid.bytes );
    }

    // get all the locally excluded lwswitch information
    FMExcludedLWSwitchInfoList excludedLwswitchInfoList;
    FMExcludedLWSwitchInfoList::iterator blit;

    mpLfm->getExcludedLwswitchInfo( excludedLwswitchInfoList );
    for (blit = excludedLwswitchInfoList.begin() ; blit != excludedLwswitchInfoList.end(); ++blit) {
        FMExcludedLWSwitchInfo_t switchInfo = *blit;
        lwswitch::lwswitchDeviceInfoMsg *devInfoMsg = switchInfoRsp->add_excludedswitchinfo();
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        devInfoMsg->set_physicalid(switchInfo.physicalId);
        pciInfo->set_domain( switchInfo.pciInfo.domain );
        pciInfo->set_bus( switchInfo.pciInfo.bus );
        pciInfo->set_device( switchInfo.pciInfo.device );
        pciInfo->set_function( switchInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        devInfoMsg->set_uuid( switchInfo.uuid.bytes );
        devInfoMsg->set_excludedreason(switchInfo.excludedReason);
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
#endif // LW_MODS_GDM_BUILD
}

void
LocalFMDevInfoMsgHdlr::onGpuDevInfoReq(lwswitch::fmMessage *pFmMessage)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    FMGpuInfoList gpuInfoList;
    FMGpuInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::gpuDeviceInfoRsp *gpuInfoRsp = new lwswitch::gpuDeviceInfoRsp();

    // get all the locally detected gpu information
    gpuInfoList.clear();
    mpLfm->getAllGpuInfo( gpuInfoList);

    for (it = gpuInfoList.begin() ; it != gpuInfoList.end(); ++it) {
        FMGpuInfo_t gpuInfo = *it;
        lwswitch::gpuDeviceInfoMsg *devInfoMsg = gpuInfoRsp->add_gpuinfo();
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        pciInfo->set_domain( gpuInfo.pciInfo.domain );
        pciInfo->set_bus( gpuInfo.pciInfo.bus );
        pciInfo->set_device( gpuInfo.pciInfo.device );
        pciInfo->set_function( gpuInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        devInfoMsg->set_gpuindex( gpuInfo.gpuIndex );
        devInfoMsg->set_uuid( gpuInfo.uuid.bytes );
        devInfoMsg->set_discoveredlinkmask( gpuInfo.discoveredLinkMask );
        devInfoMsg->set_enabledlinkmask( gpuInfo.enabledLinkMask );
        devInfoMsg->set_archtype( gpuInfo.archType );
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        devInfoMsg->set_isegmcapable( gpuInfo.isEgmCapable );
        devInfoMsg->set_isspacapable( gpuInfo.isSpaCapable );
        if ( gpuInfo.isSpaCapable ) {
            devInfoMsg->set_spaaddress( gpuInfo.spaAddress );
        }
#endif
    }
    // get all the locally excluded gpu information
    FMExcludedGpuInfoList excludedGpuInfoList;
    FMExcludedGpuInfoList::iterator blit;

    mpLfm->getExcludedGpuInfo( excludedGpuInfoList );
    for (blit = excludedGpuInfoList.begin() ; blit != excludedGpuInfoList.end(); ++blit) {
        FMExcludedGpuInfo_t gpuInfo = *blit;
        lwswitch::gpuDeviceInfoMsg *devInfoMsg = gpuInfoRsp->add_excludedgpuinfo();
        lwswitch::devicePciInfo *pciInfo = new lwswitch::devicePciInfo();
        pciInfo->set_domain( gpuInfo.pciInfo.domain );
        pciInfo->set_bus( gpuInfo.pciInfo.bus );
        pciInfo->set_device( gpuInfo.pciInfo.device );
        pciInfo->set_function( gpuInfo.pciInfo.function );
        devInfoMsg->set_allocated_pciinfo( pciInfo );
        //devInfoMsg->set_gpuindex( gpuInfo.gpuIndex );
        devInfoMsg->set_uuid( gpuInfo.uuid.bytes );
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
#endif // LW_MODS_GDM_BUILD
}

void
LocalFMDevInfoMsgHdlr::onGpuLWLinkSpeedInfoReq(lwswitch::fmMessage *pFmMessage)
{
    // MODS GDM build does not require access to HW and LFM
    // This resolves any LFM and LwSwitch Driver dependency
#if defined(LW_MODS_GDM_BUILD)
    return;
#else
    FMGpuInfoList gpuInfoList;
    FMGpuInfoList::iterator it;
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();

    // send the response back to GFM
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::gpuLWLinkSpeedInfoRsp *gpuLinkSpeedRsp = new lwswitch::gpuLWLinkSpeedInfoRsp();

    // get all the locally detected gpu information
    gpuInfoList.clear();
    mpLfm->getAllGpuInfo( gpuInfoList );

    for ( it = gpuInfoList.begin() ; it != gpuInfoList.end(); ++it ) {
        FMGpuInfo_t gpuInfo = *it;
        FMLWLinkSpeedInfoList linkSpeedInfo;
        linkSpeedInfo.clear();
        mpLfm->mFMGpuMgr->getGpuLWLinkSpeedInfo( gpuInfo.uuid, linkSpeedInfo );
        lwswitch::gpuLWLinkSpeedInfoMsg *gpuSpeedMsg = gpuLinkSpeedRsp->add_gpulinkspeedinfo();
        gpuSpeedMsg->set_uuid( gpuInfo.uuid.bytes );
        FMLWLinkSpeedInfoList::iterator jit;
        for ( jit = linkSpeedInfo.begin() ; jit != linkSpeedInfo.end(); ++jit ) {
            FMLWLinkSpeedInfo tempSpeedInfo = *jit;
            lwswitch::lwLinkSpeedInfoMsg *infoMsg = gpuSpeedMsg->add_speedinfo();
            infoMsg->set_linkindex( tempSpeedInfo.linkIndex );
            infoMsg->set_linklineratembps( tempSpeedInfo.linkLineRateMBps );
            infoMsg->set_linkclockmhz( tempSpeedInfo.linkClockMhz );
            infoMsg->set_linkclocktype( tempSpeedInfo.linkClockType );
            infoMsg->set_linkdataratekibps( tempSpeedInfo.linkDataRateKiBps );
        }
    }

    // create the final response message
    lwswitch::deviceInfoResponseMsg *rspMsg = new lwswitch::deviceInfoResponseMsg();
    rspMsg->set_inforeqid( reqMsg.inforeqid() );
    rspMsg->set_allocated_gpulinkspeedrsp( gpuLinkSpeedRsp );

    // fill the fabric message
    pFmRspMessage->set_type( lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_RSP );
    pFmRspMessage->set_allocated_devinforsp( rspMsg );
    pFmRspMessage->set_requestid( pFmMessage->requestid() );

    // send the message to GFM
    mCtrlConnIntf->SendMessageToGfm( pFmRspMessage, false );

    // free the allocated message and return final status
    delete( pFmRspMessage );
#endif // LW_MODS_GDM_BUILD
}

void
LocalFMDevInfoMsgHdlr::onGetFMVersionInfoReq(lwswitch::fmMessage *pFmMessage)
{
    lwswitch::deviceInfoRequestMsg reqMsg = pFmMessage->devinforeq();
    lwswitch::fmMessage *pFmRspMessage = new lwswitch::fmMessage();
    lwswitch::nodeVersionInfoRsp *versionInfoRsp = new lwswitch::nodeVersionInfoRsp();

    pFmRspMessage->set_requestid( pFmMessage->requestid() );

    lwswitch::deviceInfoResponseMsg *rspMsg = new lwswitch::deviceInfoResponseMsg();
    rspMsg->set_inforeqid( reqMsg.inforeqid() );
    rspMsg->set_allocated_versioninforsp( versionInfoRsp );

    //fill version string
    versionInfoRsp->set_versionstring(FM_VERSION_STRING);
    pFmRspMessage->set_allocated_devinforsp( rspMsg );
    pFmRspMessage->set_type( lwswitch::FM_NODE_GET_VERSION_INFO_RSP );

    // send the message to GFM
    mCtrlConnIntf->SendMessageToGfm( pFmRspMessage, false );

    // free the allocated message and return final status
    delete( pFmRspMessage );
}

void
LocalFMDevInfoMsgHdlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    // nothing specific for LFM to do in case of node disconnection
}

GlobalFMDevInfoMsgHdlr::GlobalFMDevInfoMsgHdlr(FMConnInterface *ctrlConnIntf)
    :FMDevInfoMsgHdlrBase( ctrlConnIntf )
{
    mDevInfoReqPending.clear();
    mDevInfoReqComplete.clear();
    lwosInitializeCriticalSection( &mLock );
    mNextInfoReqId = 0; //valid IDs start from 1
}

GlobalFMDevInfoMsgHdlr::~GlobalFMDevInfoMsgHdlr()
{
    mDevInfoReqPending.clear();
    mDevInfoReqComplete.clear();
    lwosDeleteCriticalSection( &mLock );
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendLWLinkDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_REQ, nodeId, requestId);
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendLWSwitchDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_REQ, nodeId, requestId);
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendGpuDevInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_REQ, nodeId, requestId);
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendFMVersionInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_VERSION_INFO_REQ, nodeId, requestId);
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendGpuLWLinkSpeedInfoReq(uint32 nodeId, uint64 &requestId)
{
    return sendDevInfoRequest(lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_REQ, nodeId, requestId);
}

FMIntReturn_t
GlobalFMDevInfoMsgHdlr::sendDevInfoRequest(lwswitch::FabricManagerMessageType msgType,
                                      uint32 nodeId, uint64 &requestId)
{
    FMIntReturn_t retVal;
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
        case lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_REQ: {
            lwswitch::gpuLWLinkSpeedInfoReq *gpuSpeedReq = new lwswitch::gpuLWLinkSpeedInfoReq();
            reqMsg->set_allocated_gpulinkspeedreq(gpuSpeedReq);
            break;
        }
        case lwswitch::FM_NODE_GET_VERSION_INFO_REQ: {
            lwswitch::nodeVersionInfoReq *versionInfoReq = new lwswitch::nodeVersionInfoReq();
            reqMsg->set_allocated_versioninforeq(versionInfoReq);
            break;
        }
        default:{
            FM_LOG_ERROR( "device info message handler: found unknown request type while parsing response message\n" );
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

    if ( retVal != FM_INT_ST_OK ) {
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
GlobalFMDevInfoMsgHdlr::isDevInfoReqComplete(uint64 &requestId, DevInfoReqResult &reqResult)
{
    FMAutoLock lock(mLock);
    DevInfoRequestMap::iterator it;
    it = mDevInfoReqComplete.find( requestId );
    if ( it != mDevInfoReqComplete.end() ) {
        FmDevInfoReqCtx reqCtx = it->second;
        reqResult = reqCtx.result;
        mDevInfoReqComplete.erase( requestId );
        return true;
    }

    return false;
}

void
GlobalFMDevInfoMsgHdlr::onLWLinkDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // an LWLink device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GlobalFMDevInfoMsgHdlr::onLWSwitchDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // an LWSwitch device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GlobalFMDevInfoMsgHdlr::onGpuDevInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // GPU device information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GlobalFMDevInfoMsgHdlr::onGpuLWLinkSpeedInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // GPU LWLink Speed information request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GlobalFMDevInfoMsgHdlr::onGetFMVersionInfoRsp(lwswitch::fmMessage *pFmMessage)
{
    // Get node FM version info request completed
    handleDevInfoReqCompletion(pFmMessage);
}

void
GlobalFMDevInfoMsgHdlr::handleDevInfoReqCompletion(lwswitch::fmMessage *pFmMessage)
{
    lwswitch::deviceInfoResponseMsg devInfoRsp = pFmMessage->devinforsp();
    uint64 infoReqId = devInfoRsp.inforeqid();

    // find the request locally and update its result
    // also move the request to completed list from pending
    FmDevInfoReqCtx reqCtx = {0};
    if ( !getPendingInfoReq(infoReqId, reqCtx) ) {
        FM_LOG_WARNING( "device info message handler: request id %llu not found during response message handling\n",
                        infoReqId );
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
GlobalFMDevInfoMsgHdlr::handleEvent(FabricManagerCommEventType eventType, uint32 nodeId)
{
    FM_LOG_DEBUG( "GlobalFMDevInfoMsgHdlr handleEvent\n" );

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
GlobalFMDevInfoMsgHdlr::addToReqPendingTable(uint64 reqId, uint32 toNodeId)
{
    FMAutoLock lock(mLock);

    // keep the request information locally and track
    FmDevInfoReqCtx reqCtx;
    memset( &reqCtx, 0, sizeof(reqCtx) );

    reqCtx.toNodeId = toNodeId;

    if ( mDevInfoReqPending.count(reqId) ) {
        FM_LOG_WARNING( "device info message handler: request id %llu already exists in pending request list\n",
                        reqId );
        return;
    }

    mDevInfoReqPending.insert( std::make_pair(reqId, reqCtx) );
}

void
GlobalFMDevInfoMsgHdlr::removeFromReqPendingTable(uint64 reqId)
{
    FMAutoLock lock(mLock);

    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it != mDevInfoReqPending.end() ) {
        mDevInfoReqPending.erase( it );
    }
}

bool
GlobalFMDevInfoMsgHdlr::getPendingInfoReq(uint64 reqId, FmDevInfoReqCtx &reqCtx)
{
    FMAutoLock lock(mLock);

    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it == mDevInfoReqPending.end() ) {
        return false;
    }

    // found the request, copy it and return success
    reqCtx = it->second;
    return true;
}

void
GlobalFMDevInfoMsgHdlr::markPendingReqAsComplete(uint64 reqId, FmDevInfoReqCtx &reqCtx)
{
    FMAutoLock lock(mLock);
    DevInfoRequestMap::iterator it = mDevInfoReqPending.find( reqId );
    if ( it == mDevInfoReqPending.end() ) {
        return;
    }

    // erase from pending and add to complete list
    mDevInfoReqPending.erase( it );
    mDevInfoReqComplete.insert( std::make_pair(reqId, reqCtx) );
}

uint64
GlobalFMDevInfoMsgHdlr::getNextDevInfoReqId(void)
{
    FM_LOG_DEBUG( "GlobalFMDevInfoMsgHdlr getNextDevInfoReqId\n" );
    FMAutoLock lock(mLock);

    mNextInfoReqId++;
    if ( mNextInfoReqId == 0 ) {
        // wrap around
        mNextInfoReqId = 1;
    }
    return mNextInfoReqId;
}

void
GlobalFMDevInfoMsgHdlr::handleNodeDisconnect(uint32 nodeId)
{
    // node's socket connection got disconnected. complete all the
    // outstanding request to this LFM node as completed with
    // appropriate error code.
    FMAutoLock lock(mLock);

    FM_LOG_DEBUG( "GlobalFMDevInfoMsgHdlr handleNodeDisconnect\n" );

    DevInfoRequestMap::iterator it = mDevInfoReqPending.begin();
    while ( it != mDevInfoReqPending.end() ) {
        FmDevInfoReqCtx reqCtx = it->second;
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
