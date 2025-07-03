#include <string.h>

#include "DcgmLocalFabricManager.h"
#include "logging.h"
#include "DcgmFMLWLinkReqInit.h"
#include "DcgmFMLWLinkError.h"
#include <g_lwconfig.h>


DcgmFMLWLinkReqInit::DcgmFMLWLinkReqInit(lwswitch::fmMessage *pFmMessage,
                                         FMConnInterface *ctrlConnIntf,
                                         DcgmFMLWLinkDrvIntf *linkDrvIntf,
                                         DcgmLFMLWLinkDevRepo *linkDevRepo)
    :DcgmFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqInit::DcgmFMLWLinkReqInit \n" );
}

DcgmFMLWLinkReqInit::~DcgmFMLWLinkReqInit()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqInit::~DcgmFMLWLinkReqInit \n" );
}

bool
DcgmFMLWLinkReqInit::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqInit processNewMasterRequest\n" );

    // initialization requests don't have master/slave LFM sync
    // so they complete from this context itself.
    bool bReqCompleted = true;    

    // get exclusive access to our request object
    lock();

    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE: {
            doEnableTxCommonMode();
            break;
        }
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE: {
            doDisableTxCommonMode();
            break;
        }
        case lwswitch::FM_LWLINK_CALIBRATE: {
            doCalibrate();
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_DATA: {
            doEnableData();
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1: {
            doInitphase1();
            break;
        }
        case lwswitch::FM_LWLINK_RX_INIT_TERM: {
            doRxInitTerm();
            break;
        }
        case lwswitch::FM_LWLINK_SET_RX_DETECT: {
            doSetRxDetect();
            break;
        }
        case lwswitch::FM_LWLINK_GET_RX_DETECT: {
            doGetRxDetect();
            break;
        }
        case lwswitch::FM_LWLINK_INITNEGOTIATE: {
            doInitnegotiate();
            break;
        }
#endif
        case lwswitch::FM_LWLINK_INIT: {
            doLinkInit();
            break;
        }
        case lwswitch::FM_LWLINK_INIT_STATUS: {
            doLinkInitStatus();
            break;
        }
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS: {
            doResetSwitchLinks( pFmMessage->lwlinkmsg() );
            break;
        }
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS: {
            doResetAllSwitchLinks( pFmMessage->lwlinkmsg() );
            break;
        }
        default:{
            PRINT_ERROR( "", "Unknown Node Init request type\n" );
            break;
        }
    }

    // the request is finished. send response to global FM
    sendRequestCompletion();
    unLock();
    return bReqCompleted;
}

bool
DcgmFMLWLinkReqInit::processReqTimeOut()
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqInit processReqTimeOut\n" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock( );
    return bReqCompleted;
}

void
DcgmFMLWLinkReqInit::doEnableTxCommonMode(void)
{
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = true;
    // enable common mode ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(modeParam.status) );
}

void
DcgmFMLWLinkReqInit::doDisableTxCommonMode(void)
{
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = false;
    // disable common mode ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(modeParam.status) );
}

void
DcgmFMLWLinkReqInit::doCalibrate(void)
{
    lwlink_calibrate calibrateParam = {0};
    // RX Calibration ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_CALIBRATE, &calibrateParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(calibrateParam.status) );
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
DcgmFMLWLinkReqInit::doInitphase1(void)
{
    lwlink_initphase1 initphase1Param = {0};
    // This is done by RM in parallel for all links. Because of this parallel 
    // exelwtion, it shouldn't take long for the the entire node.
    // TODO: revisit once hardware is available and document how much time is taken 
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_INITPHASE1, &initphase1Param );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(initphase1Param.status) );
}

void
DcgmFMLWLinkReqInit::doRxInitTerm(void)
{
    lwlink_rx_init_term rxInitTermParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_RX_INIT_TERM, &rxInitTermParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(rxInitTermParam.status) );
}

void
DcgmFMLWLinkReqInit::doSetRxDetect(void)
{
    lwlink_set_rx_detect setRxDetectParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_RX_DETECT, &setRxDetectParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(setRxDetectParam.status) );
}

void
DcgmFMLWLinkReqInit::doGetRxDetect(void)
{
    lwlink_get_rx_detect getRxDetectParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_GET_RX_DETECT, &getRxDetectParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(getRxDetectParam.status) );
}
#endif

void
DcgmFMLWLinkReqInit::doEnableData(void)
{
    lwlink_enable_data enableDataParam = {0};
    // enable data ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_ENABLE_DATA, &enableDataParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(enableDataParam.status) );
}

void
DcgmFMLWLinkReqInit::doLinkInit(void)
{
    lwlink_link_init_async linkInitParam = {0};
    // link init ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC, &linkInitParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode(linkInitParam.status) );
}

void
DcgmFMLWLinkReqInit::doLinkInitStatus(void)
{
    DcgmFMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    DcgmFMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;

    /*
     * TODO/NOTE: Link init status may be time sensitive as the driver
     * has to wait (poll) for each link state to change. Using a per device
     * thread to send the INIT ioctl will improve the overall exelwtion time.
     */

    // clean any cached token list before starting new one
    mInitStatusList.clear();

    // link init status ioctl is per device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_device_link_init_status statusParam;
        memset( &statusParam, 0, sizeof(statusParam));
        DcgmFMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        statusParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        statusParam.devInfo.nodeId  = mLWLinkDevRepo->getLocalNodeId();

        mLWLinkDrvIntf->doIoctl( IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS, &statusParam );
        if ( statusParam.status != LWL_SUCCESS ) {
            status = DcgmFMLWLinkError::getLinkErrorCode( statusParam.status );
            // clear any outstanding status info as the request is marked as failed
            mInitStatusList.clear();
            break;
        }
        // copy the link init status information
        DcgmLinkInitStatusInfo dcgmStatusInfo;
        uint64 deviceId = lwlinkDevInfo.getDeviceId();
        dcgmStatusInfo.nodeId= mLWLinkDevRepo->getLocalNodeId();
        dcgmStatusInfo.gpuOrSwitchId = deviceId;
        memcpy( dcgmStatusInfo.initStatus, statusParam.linkStatus, sizeof(statusParam.linkStatus) );
        mInitStatusList.push_back( dcgmStatusInfo );
        // also update our local device repo information
        mLWLinkDevRepo->setDevLinkInitStatus(deviceId, statusParam.linkStatus);
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
DcgmFMLWLinkReqInit::doInitnegotiate(void)
{
    lwlink_initnegotiate initNegotiateParam;;
    memset(&initNegotiateParam, 0, sizeof(initNegotiateParam));

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_INITNEGOTIATE, &initNegotiateParam );
    setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode( initNegotiateParam.status ) );

}
#endif

void
DcgmFMLWLinkReqInit::doResetSwitchLinks(const lwswitch::lwlinkMsg &linkMsg)
{
    const lwswitch::lwlinkRequestMsg &linkReqMsg = linkMsg.reqmsg();
    const lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg &resetReqMsg = linkReqMsg.nodeinitresetswitchlinksreqmsg();
    int status = FM_LWLINK_ST_SUCCESS;

    //
    // link reset is implemented by individual endpoint drivers, ie not by 
    // LWLinkCoreLib driver. So issue ioctl through LWSwitch driver interface
    //

    DcgmLocalFabricManagerControl *pLfmControl = (DcgmLocalFabricManagerControl*)mCtrlConnIntf;
    DcgmSwitchInterface *pSwitchInterface;
    pSwitchInterface = pLfmControl->switchInterfaceAt( resetReqMsg.switchphysicalid() );
    if ( !pSwitchInterface ) {
        PRINT_ERROR( "%d", "Invalid switch driver interface for physical Id %d during link reset.",
                    resetReqMsg.switchphysicalid() );

        setCompletionStatus( DcgmFMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND ) );
        return;
    }

    // issue the switch ioctl
    switchIoctl_t ioctlStruct;
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS ioctlParams;
    ioctlParams.linkMask = resetReqMsg.linkmask();

    // construct the ioctl placeholder
    ioctlStruct.type = IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS;
    ioctlStruct.ioctlParams = &ioctlParams;

    FM_ERROR_CODE ret = pSwitchInterface->doIoctl( &ioctlStruct );
    if ( ret != FM_SUCCESS ) {
        status = DcgmFMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
    }
    
    // update the final ioctl status
    setCompletionStatus( status );
}

void
DcgmFMLWLinkReqInit::doResetAllSwitchLinks(const lwswitch::lwlinkMsg &linkMsg)
{
    DcgmFMLWSwitchInfoList switchInfoList;
    DcgmFMLWSwitchInfoList::iterator it;
    DcgmLocalFabricManagerControl *pLfmControl = (DcgmLocalFabricManagerControl*)mCtrlConnIntf;
    int status = FM_LWLINK_ST_SUCCESS;

    //
    // link reset is implemented by individual endpoint drivers, ie not by 
    // LWLinkCoreLib driver. So issue ioctl through LWSwitch driver interface
    //

    // get all the locally detected switch details to reset their links.
    pLfmControl->getAllLwswitchInfo( switchInfoList );
    for (it = switchInfoList.begin() ; it != switchInfoList.end(); ++it) {
        DcgmFMLWSwitchInfo switchInfo = *it;
        DcgmSwitchInterface *pSwitchInterface = NULL;
        pSwitchInterface = pLfmControl->switchInterfaceAt( switchInfo.physicalId );
        if ( !pSwitchInterface ) {
            PRINT_ERROR( "%d", "Invalid switch driver interface for physical Id %d during link reset.",
                        switchInfo.physicalId );
            // continue with rest of the switches, but mark overall status as failed
            status = DcgmFMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND );
            continue;
        }

        // reset all the enabled links of the switch
        // reset should be in pairs. so compute the mask considering the odd/even pair
        uint64 tempEnabledMask = switchInfo.enabledLinkMask;
        uint64 resetLinkMask = 0;
        for (uint64_t linkId = 0; tempEnabledMask != 0; linkId +=2, tempEnabledMask >>= 2) {
            if ((tempEnabledMask & 0x3) != 0) {
                // rebuild the mask
                resetLinkMask |= (BIT64(linkId) | BIT64(linkId + 1));
            }
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS ioctlParams;
        ioctlParams.linkMask = resetLinkMask;
        // construct the ioctl placeholder
        ioctlStruct.type = IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS;
        ioctlStruct.ioctlParams = &ioctlParams;

        FM_ERROR_CODE ret = pSwitchInterface->doIoctl( &ioctlStruct );
        if ( ret != FM_SUCCESS ) {
            // continue with rest of the switches, but mark overall status as failed
            status = DcgmFMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
DcgmFMLWLinkReqInit::sendRequestCompletion(void)
{
    PRINT_DEBUG( "", "DcgmFMLWLinkReqInit sendRequestCompletion\n" );

    // send the response to GFM
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    dcgmReturn_t retVal;
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();

    // fill the response msg based on the actual request from GFM
    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE: {
            genEnableCommonModeResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE: {
            genDisableCommonModeResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_CALIBRATE: {
            genCalibrateResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_ENABLE_DATA: {
            genEnableDataResp( rspMsg );
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_LWLINK_INITPHASE1: {
            genInitphase1Resp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_RX_INIT_TERM: {
            genRxInitTermResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_SET_RX_DETECT: {
            genSetRxDetectResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_GET_RX_DETECT: {
            genGetRxDetectResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_INITNEGOTIATE: {
            genInitnegotiateResp( rspMsg );
            break;
        }
#endif
        case lwswitch::FM_LWLINK_INIT: {
            genLinkInitResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_INIT_STATUS: {
            genLinkInitStatusResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_RESET_SWITCH_LINKS: {
            genResetSwitchLinksResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_RESET_ALL_SWITCH_LINKS: {
            genResetAllSwitchLinksResp( rspMsg );
            break;
        }        
        default:{
            PRINT_ERROR( "", "Unknown Node Init request type while preparing response\n" );
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
DcgmFMLWLinkReqInit::genEnableCommonModeResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genDisableCommonModeResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genCalibrateResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );

}

void
DcgmFMLWLinkReqInit::genEnableDataResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
DcgmFMLWLinkReqInit::genInitphase1Resp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genRxInitTermResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genSetRxDetectResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genGetRxDetectResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genInitnegotiateResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}
#endif

void
DcgmFMLWLinkReqInit::genLinkInitResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
DcgmFMLWLinkReqInit::genLinkInitStatusResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkNodeInitStatusRspMsg *statusRspMsg = new lwswitch::lwlinkNodeInitStatusRspMsg();
    DcgmLinkInitStatusInfoList::iterator it;

    // copy the link status information to global FM.
    for ( it = mInitStatusList.begin(); it != mInitStatusList.end(); it++ ) {
        // copy device status information
        DcgmLinkInitStatusInfo dcgmStatusInfo = *it;
        lwswitch::lwlinkDeviceLinkInitStatus *devStatusMsg = statusRspMsg->add_initstatus();
        devStatusMsg->set_gpuorswitchid( dcgmStatusInfo.gpuOrSwitchId );
        // copy per link status information of the device
        for ( int idx = 0; idx < LWLINK_MAX_DEVICE_CONN; idx++ ) {
            lwswitch::lwlinkLinkInitStatus *linkStatus = devStatusMsg->add_linkstatus();
            linkStatus->set_linkindex( dcgmStatusInfo.initStatus[idx].linkIndex );
            linkStatus->set_status( dcgmStatusInfo.initStatus[idx].initStatus );
        }
    }

    statusRspMsg->set_nodeid( mLWLinkDevRepo->getLocalNodeId() );
    rspMsg->set_allocated_nodeinitstatusrspmsg( statusRspMsg );
}

void
DcgmFMLWLinkReqInit::genResetSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // no request specific data to return other than the overall status, which is part of the
    // parent lwlinkResponseMsg. Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitResetSwitchLinksRspMsg *resetRspMsg = new lwswitch::lwlinkNodeInitResetSwitchLinksRspMsg();
    rspMsg->set_allocated_nodeinitresetswitchlinksrspmsg( resetRspMsg );
}

void
DcgmFMLWLinkReqInit::genResetAllSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // no request specific data to return other than the overall status, which is part of the
    // parent lwlinkResponseMsg. Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitResetAllSwitchLinksRspMsg *resetRspMsg = new lwswitch::lwlinkNodeInitResetAllSwitchLinksRspMsg();
    rspMsg->set_allocated_nodeinitresetallswitchlinksrspmsg( resetRspMsg );
}

void
DcgmFMLWLinkReqInit::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Link Node Init Req Information" << std::endl;

    // append base request dump information
    DcgmFMLWLinkReqBase::dumpInfo( os );
}
