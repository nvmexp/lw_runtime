/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <string.h>

#include "LocalFabricManager.h"
#include "fm_log.h"
#include "LocalFMLWLinkReqInit.h"
#include "FMLWLinkError.h"
#include <g_lwconfig.h>
#include "timelib.h"

LocalFMLWLinkReqInit::LocalFMLWLinkReqInit(lwswitch::fmMessage *pFmMessage,
                                           FMConnInterface *ctrlConnIntf,
                                           LocalFMLWLinkDrvIntf *linkDrvIntf,
                                           LocalFMLWLinkDevRepo *linkDevRepo)
    :LocalFMLWLinkReqBase( pFmMessage, ctrlConnIntf, linkDrvIntf, linkDevRepo)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqInit::LocalFMLWLinkReqInit" );
}

LocalFMLWLinkReqInit::~LocalFMLWLinkReqInit()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqInit::~LocalFMLWLinkReqInit" );
}

bool
LocalFMLWLinkReqInit::processNewMasterRequest(lwswitch::fmMessage *pFmMessage)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqInit processNewMasterRequest" );

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
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS: {
            doOpticalInitLinks();
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST: {
            doOpticalEnableIobist();
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX: {
            doOpticalStartPretrain(true);
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX: {
            doOpticalCheckPretrain(true);
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX: {
            doOpticalStartPretrain(false);
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX: {
            doOpticalCheckPretrain(false);
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN: {
            doOpticalStopPretrain();
            break;
        }
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST: {
            doOpticalDisableIobist();
            break;
        }
#endif
        case lwswitch::FM_LWLINK_INITPHASE1: {
            doInitphase1();
            break;
        }
        case lwswitch::FM_LWLINK_INITPHASE5: {
            doInitphase5();
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
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO: {
            doSwitchTrainingFailedLinkInfo( pFmMessage->lwlinkmsg() );
            break;
        }
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE: {
            doGetDeviceLwlinkState();
            break;
        }
        default:{
            FM_LOG_ERROR( "unknown request type %d detected in LWLink initialization handler",
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
LocalFMLWLinkReqInit::processReqTimeOut()
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqInit processReqTimeOut" );
    bool bReqCompleted = true;

    // get exclusive access to our request object
    lock();

    setCompletionStatus( FM_LWLINK_ST_TIMEOUT );
    sendRequestCompletion();

    unLock( );
    return bReqCompleted;
}

void
LocalFMLWLinkReqInit::doEnableTxCommonMode(void)
{
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = true;
    // enable common mode ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam, sizeof(modeParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(modeParam.status) );
}

void
LocalFMLWLinkReqInit::doDisableTxCommonMode(void)
{
    lwlink_set_tx_common_mode modeParam;
    modeParam.commMode = false;
    // disable common mode ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_TX_COMMON_MODE, &modeParam, sizeof(modeParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(modeParam.status) );
}

void
LocalFMLWLinkReqInit::doCalibrate(void)
{
    lwlink_calibrate calibrateParam = {0};
    // RX Calibration ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_CALIBRATE, &calibrateParam, sizeof(calibrateParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(calibrateParam.status) );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
LocalFMLWLinkReqInit::doOpticalInitLinks(void)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_INIT_LINKS");
    lwlink_optical_init_links param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_INIT_LINKS, &param, sizeof(param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
}

void
LocalFMLWLinkReqInit::doOpticalEnableIobist(void)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_ENABLE_IOBIST");
    
    lwlink_optical_set_iobist param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_ENABLE_IOBIST, &param, sizeof(param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
    
}

void
LocalFMLWLinkReqInit::doOpticalStartPretrain(bool isTx)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_START_PRETRAIN TX=%s", isTx? "true": "false");
    lwlink_optical_set_pretrain param = {0};
    param.bTx = isTx;

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_START_PRETRAIN, &param, sizeof(param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
}

void
LocalFMLWLinkReqInit::doOpticalCheckPretrain(bool isTx)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_CHECK_PRETRAIN TX=%s", isTx? "true": "false");
    lwlink_optical_check_pretrain param = {0};
    param.bTx = isTx;

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_CHECK_PRETRAIN, &param, sizeof(param) );

    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
}

void
LocalFMLWLinkReqInit::doOpticalStopPretrain(void)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_STOP_PRETRAIN");
    lwlink_optical_set_pretrain param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_STOP_PRETRAIN, &param, sizeof(param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
}

void
LocalFMLWLinkReqInit::doOpticalDisableIobist(void)
{
    FM_LOG_DEBUG("IOCTL_LWLINK_OPTICAL_DISABLE_IOBIST");
    lwlink_optical_set_iobist param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_OPTICAL_DISABLE_IOBIST, &param, sizeof(param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(param.status) );
}
#endif

void
LocalFMLWLinkReqInit::doInitphase1(void)
{
    lwlink_initphase1 initphase1Param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_INITPHASE1, &initphase1Param, sizeof(initphase1Param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(initphase1Param.status) );
}

  void
LocalFMLWLinkReqInit::doRxInitTerm(void)
{
    lwlink_rx_init_term rxInitTermParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_RX_INIT_TERM, &rxInitTermParam, sizeof(rxInitTermParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(rxInitTermParam.status) );
}

void
LocalFMLWLinkReqInit::doSetRxDetect(void)
{
    lwlink_set_rx_detect setRxDetectParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_SET_RX_DETECT, &setRxDetectParam, sizeof(setRxDetectParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(setRxDetectParam.status) );
}

void
LocalFMLWLinkReqInit::doGetRxDetect(void)
{
    lwlink_get_rx_detect getRxDetectParam = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_GET_RX_DETECT, &getRxDetectParam, sizeof(getRxDetectParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(getRxDetectParam.status) );
}

void
LocalFMLWLinkReqInit::doEnableData(void)
{
    lwlink_enable_data enableDataParam = {0};
    // enable data ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_ENABLE_DATA, &enableDataParam, sizeof(enableDataParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(enableDataParam.status) );
}

void
LocalFMLWLinkReqInit::doInitphase5(void)
{
    lwlink_initphase5 initphase5Param = {0};

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_INITPHASE5, &initphase5Param, sizeof(initphase5Param) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(initphase5Param.status) );
}

void
LocalFMLWLinkReqInit::doLinkInit(void)
{
    lwlink_link_init_async linkInitParam = {0};
    // link init ioctl is for whole node and it shouldn't take much time 
    // as it simply writing only few registers, i.e. no link state change polling.
    mLWLinkDrvIntf->doIoctl( IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC, &linkInitParam, sizeof(linkInitParam) );
    setCompletionStatus( FMLWLinkError::getLinkErrorCode(linkInitParam.status) );
}

void
LocalFMLWLinkReqInit::doLinkInitStatus(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
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
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        // fill the LWLink Device information
        statusParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();
        statusParam.devInfo.nodeId  = mLWLinkDevRepo->getLocalNodeId();

        mLWLinkDrvIntf->doIoctl( IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS, &statusParam, sizeof(statusParam) );
        if ( statusParam.status != LWL_SUCCESS ) {
            status = FMLWLinkError::getLinkErrorCode( statusParam.status );
            // clear any outstanding status info as the request is marked as failed
            mInitStatusList.clear();
            break;
        }
        // copy the link init status information
        FMLinkInitStatusInfo fmStatusInfo;
        uint64 deviceId = lwlinkDevInfo.getDeviceId();
        fmStatusInfo.nodeId= mLWLinkDevRepo->getLocalNodeId();
        fmStatusInfo.gpuOrSwitchId = deviceId;
        memcpy( fmStatusInfo.initStatus, statusParam.linkStatus, sizeof(statusParam.linkStatus) );
        mInitStatusList.push_back( fmStatusInfo );
        // also update our local device repo information
        mLWLinkDevRepo->setDevLinkInitStatus(deviceId, statusParam.linkStatus);
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::doInitnegotiate(void)
{
    int status = FM_LWLINK_ST_SUCCESS;

    lwlink_initnegotiate initNegotiateParam;;
    memset(&initNegotiateParam, 0, sizeof(initNegotiateParam));

    mLWLinkDrvIntf->doIoctl( IOCTL_LWLINK_INITNEGOTIATE, &initNegotiateParam, sizeof(initNegotiateParam) );
    if(initNegotiateParam.status != LWL_SUCCESS ) {
        status = FMLWLinkError::getLinkErrorCode( initNegotiateParam.status );
        FM_LOG_DEBUG( "INITNEGOTIATE failed status=%d", initNegotiateParam.status);
    }
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::doResetSwitchLinks(const lwswitch::lwlinkMsg &linkMsg)
{
    const lwswitch::lwlinkRequestMsg &linkReqMsg = linkMsg.reqmsg();
    const lwswitch::lwlinkNodeInitResetSwitchLinksReqMsg &resetReqMsg = linkReqMsg.nodeinitresetswitchlinksreqmsg();
    int status = FM_LWLINK_ST_SUCCESS;

    //
    // link reset is implemented by individual endpoint drivers, ie not by 
    // LWLinkCoreLib driver. So issue ioctl through LWSwitch driver interface
    //

    LocalFabricManagerControl *pLfmControl = (LocalFabricManagerControl*)mCtrlConnIntf;

    // this is needed because of the trunk link reset done in partition activation/deactivation
    if ( pLfmControl->isSwitchDegraded(resetReqMsg.switchphysicalid()) ) {
        setCompletionStatus(FM_LWLINK_ST_SUCCESS);
        return;
    }

    LocalFMSwitchInterface *pSwitchInterface;
    pSwitchInterface = pLfmControl->switchInterfaceAt( resetReqMsg.switchphysicalid() );
    if ( !pSwitchInterface ) {
        FM_LOG_ERROR( "invalid switch driver interface for physical id %d during link reset.",
                    resetReqMsg.switchphysicalid() );

        setCompletionStatus( FMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND ) );
        return;
    }

    // issue the switch ioctl
    switchIoctl_t ioctlStruct;
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS ioctlParams;
    ioctlParams.linkMask = resetReqMsg.linkmask();

    // construct the ioctl placeholder
    ioctlStruct.type = IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    FM_LOG_DEBUG( "calling IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS for switch physical id %d link mask %llx",\
                  resetReqMsg.switchphysicalid(), ioctlParams.linkMask);
    FMIntReturn_t ret = pSwitchInterface->doIoctl( &ioctlStruct );
    if ( ret != FM_INT_ST_OK ) {
        status = FMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
    }
    
    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::doResetAllSwitchLinks(const lwswitch::lwlinkMsg &linkMsg)
{
    FMLWSwitchInfoList switchInfoList;
    FMLWSwitchInfoList::iterator it;
    LocalFabricManagerControl *pLfmControl = (LocalFabricManagerControl*)mCtrlConnIntf;
    int status = FM_LWLINK_ST_SUCCESS;

    //
    // link reset is implemented by individual endpoint drivers, ie not by 
    // LWLinkCoreLib driver. So issue ioctl through LWSwitch driver interface
    //

    // get all the locally detected switch details to reset their links.
    pLfmControl->getAllLwswitchInfo( switchInfoList );
    for (it = switchInfoList.begin() ; it != switchInfoList.end(); ++it) {
        FMLWSwitchInfo switchInfo = *it;

        if ( pLfmControl->isSwitchDegraded(switchInfo.physicalId) ) {
            FM_LOG_INFO("ignoring LWLink reset for LWSwitch physical id %d as it is marked as degraded", switchInfo.physicalId);
            continue;
        }

        LocalFMSwitchInterface *pSwitchInterface = NULL;
        pSwitchInterface = pLfmControl->switchInterfaceAt( switchInfo.physicalId );
        if ( !pSwitchInterface ) {
            FM_LOG_ERROR( "invalid switch driver interface for physical id %d during link reset.",
                        switchInfo.physicalId );
            // continue with rest of the switches, but mark overall status as failed
            status = FMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND );
            continue;
        }

        //
        // reset should be in pairs for Willow. so compute the mask considering the odd/even pair
        //
        uint64 resetLinkMask = 0;
        if (pSwitchInterface->getSwitchArchType() == LWSWITCH_ARCH_TYPE_SV10) {
            uint64 tempEnabledMask = switchInfo.enabledLinkMask;
            for (uint64_t linkId = 0; tempEnabledMask != 0; linkId +=2, tempEnabledMask >>= 2) {
                if ((tempEnabledMask & 0x3) != 0) {
                    // rebuild the mask
                    resetLinkMask |= (BIT64(linkId) | BIT64(linkId + 1));
                }
            }
        } else {
            //
            // in LimeRock, just reset all the enabled links, 
            //
            resetLinkMask = switchInfo.enabledLinkMask;
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS ioctlParams;
        ioctlParams.linkMask = resetLinkMask;
        // construct the ioctl placeholder
        ioctlStruct.type = IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        FMIntReturn_t ret = pSwitchInterface->doIoctl( &ioctlStruct );
        if ( ret != FM_INT_ST_OK ) {
            // continue with rest of the switches, but mark overall status as failed
            status = FMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::doSwitchTrainingFailedLinkInfo(const lwswitch::lwlinkMsg &linkMsg)
{
    const lwswitch::lwlinkRequestMsg &linkReqMsg = linkMsg.reqmsg();
    const lwswitch::lwlinkSwitchTrainingFailedReqMsg &trainingFailedReqMsg = linkReqMsg.switchtrainingfailedreqmsg();
    int status = FM_LWLINK_ST_SUCCESS;

    //
    // switch link training failed ioctl is implemented in switch driver, i.e. not through
    // LWLinkCoreLib driver. So, issue ioctl through LWSwitch driver interface
    //

    LocalFabricManagerControl *pLfmControl = (LocalFabricManagerControl*)mCtrlConnIntf;

    if ( pLfmControl->isSwitchDegraded(trainingFailedReqMsg.switchphysicalid()) ) {
        setCompletionStatus(FM_LWLINK_ST_SUCCESS);
        return;
    }

    LocalFMSwitchInterface *pSwitchInterface;
    pSwitchInterface = pLfmControl->switchInterfaceAt( trainingFailedReqMsg.switchphysicalid() );
    if ( !pSwitchInterface ) {
        FM_LOG_ERROR( "invalid switch driver interface for physical id %d while setting training failed lwlink information",
                      trainingFailedReqMsg.switchphysicalid() );

        setCompletionStatus( FMLWLinkError::getLinkErrorCode( FM_LWLINK_ST_LWL_NOT_FOUND ) );
        return;
    }

    // issue the switch ioctl
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.attemptedTrainingMask0 = trainingFailedReqMsg.trainingattemptedmask0();
    ioctlParams.trainingErrorMask0 = trainingFailedReqMsg.trainingfailedmask0();

    FMPciInfo_t pciInfo = pSwitchInterface->getSwtichPciInfo();
    FM_LOG_INFO("setting LWLink training attempted mask to 0x%llx and failed mask to 0x%llx for LWSwitch physical id %d pci bus id %s",
                 ioctlParams.attemptedTrainingMask0, ioctlParams.trainingErrorMask0,
                 trainingFailedReqMsg.switchphysicalid(), pciInfo.busId);

    // construct the ioctl placeholder
    ioctlStruct.type = IOCTL_LWSWITCH_SET_TRAINING_ERROR_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

#ifndef LW_MODS
    FMIntReturn_t ret = pSwitchInterface->doIoctl( &ioctlStruct );
    if ( ret != FM_INT_ST_OK ) {
        status = FMLWLinkError::getLinkErrorCode( LWL_BAD_ARGS );
    }
#else
    // This IOCTL uses SMBPBI, but SMBPBI is lwrrently unsupported in MODS
    // See Bug 2677575
    FM_LOG_INFO("IOCTL_LWSWITCH_SET_TRAINING_ERROR_INFO Unsupported!\n");
#endif

    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::doGetDeviceLwlinkState(void)
{
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;
    int status = FM_LWLINK_ST_SUCCESS;

    // send ali status ioctl to each device, so iterate for each device
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        lwlink_get_device_link_states getDeviceLinkStateParam;
        memset(&getDeviceLinkStateParam, 0, sizeof(getDeviceLinkStateParam));
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        const unsigned int TIMEOUT = 10; // 10 seconds
        timelib64_t timeStart = timelib_usecSince1970();
        timelib64_t timeNow = timeStart;
        bool doIoctlAgain = false;

        // fill the LWLink Device information
        getDeviceLinkStateParam.devInfo.nodeId = mLWLinkDevRepo->getLocalNodeId();
        getDeviceLinkStateParam.devInfo.pciInfo = lwlinkDevInfo.getDevicePCIInfo();

        while (true) {
            FM_LOG_DEBUG("Ilwoke IOCTL_LWLINK_GET_DEVICE_LINK_STATES: " NODE_ID_LOG_STR " %u PCI Device BDF %x:%x:%x:%x",
                          getDeviceLinkStateParam.devInfo.nodeId, getDeviceLinkStateParam.devInfo.pciInfo.domain,
                          getDeviceLinkStateParam.devInfo.pciInfo.bus, getDeviceLinkStateParam.devInfo.pciInfo.device,
                          getDeviceLinkStateParam.devInfo.pciInfo.function);

            // FM calls IOCTL to check device lwlink states
            mLWLinkDrvIntf->doIoctl(IOCTL_LWLINK_GET_DEVICE_LINK_STATES, &getDeviceLinkStateParam,
                                    sizeof(lwlink_get_device_link_states));

            for (uint32_t i = 0; i < getDeviceLinkStateParam.endStatesCount; i++) {
                // If the link is still training, repeat IOCTL (edge case potentially not needed)
                if ((getDeviceLinkStateParam.status != LWL_SUCCESS) ||
                    ((getDeviceLinkStateParam.endStates[i].txSubLinkMode == lwlink_tx_sublink_mode_train) &&
                    (getDeviceLinkStateParam.endStates[i].rxSubLinkMode == lwlink_rx_sublink_mode_train) &&
                    (getDeviceLinkStateParam.endStates[i].linkMode != lwlink_link_mode_active))) {
                    doIoctlAgain = true;
                    FM_LOG_DEBUG("Retry IOCTL_LWLINK_GET_DEVICE_LINK_STATES");
                    break;
                }
            }

            /* IOCTL request is completed successfully */
            if (!doIoctlAgain) {
                FM_LOG_DEBUG("IOCTL_LWLINK_GET_DEVICE_LINK_STATES is successful");
                break;
            }

            /* Repeat IOCTL if TIMEOUT has not oclwrred */
            timeNow = timelib_usecSince1970();
            if ((timeNow - timeStart) > TIMEOUT*10000000) {
                // elapsed all the time and still there are unconnected nodes.
                FM_LOG_DEBUG("Time elapsed - IOCTL_LWLINK_GET_DEVICE_LINK_STATES");
                break;
            }
        }

        if (getDeviceLinkStateParam.status != LWL_SUCCESS) {
            status = FMLWLinkError::getLinkErrorCode(getDeviceLinkStateParam.status );
            FM_LOG_DEBUG("Returned failure - IOCTL_LWLINK_GET_DEVICE_LINK_STATES");
        } else {
            // this request is completed successfully, copy the linkk state information
            // locally, which should be send to GFM as part of the reply.
            for (uint32_t i = 0; i < getDeviceLinkStateParam.endStatesCount; i++) {
                FMLWLinkStateInfo linkState;

                // Check the specific LWLink is ACTIVE
                if (getDeviceLinkStateParam.endStates[i].linkMode == lwlink_link_mode_active) {
                    linkState.linkMode = lwlink_link_mode_active;
                    linkState.txSubLinkMode = lwlink_tx_sublink_mode_hs;
                    linkState.rxSubLinkMode = lwlink_tx_sublink_mode_hs;
                    FM_LOG_DEBUG("Returned success with status - IOCTL_LWLINK_GET_DEVICE_LINK_STATES - ACTIVE");
                    // Check the specific LWLink is FAULTY
                } else if ((getDeviceLinkStateParam.endStates[i].linkMode == lwlink_link_mode_fail) ||
                           (getDeviceLinkStateParam.endStates[i].linkMode == lwlink_link_mode_fault)) {
                    linkState.linkMode = lwlink_link_mode_fault;
                    linkState.txSubLinkMode = lwlink_tx_sublink_mode_off;
                    linkState.rxSubLinkMode = lwlink_tx_sublink_mode_off;
                    FM_LOG_DEBUG("Returned success with status - IOCTL_LWLINK_GET_DEVICE_LINK_STATES - FAULTY");
                } else {
                    // Check the specific LWLink is UNKNOWN
                    linkState.linkMode = lwlink_link_mode_unknown;
                    linkState.txSubLinkMode = lwlink_tx_sublink_mode_unknown;
                    linkState.rxSubLinkMode = lwlink_tx_sublink_mode_unknown;
                    FM_LOG_DEBUG("Returned success with status - IOCTL_LWLINK_GET_DEVICE_LINK_STATES - UNKNOWN");
                }

                // Set linkstate
                lwlinkDevInfo.setLinkState(i, linkState);
            }
        }
    }

    // update the final ioctl status
    setCompletionStatus( status );
}

void
LocalFMLWLinkReqInit::sendRequestCompletion(void)
{
    FM_LOG_DEBUG( "LocalFMLWLinkReqInit sendRequestCompletion" );

    // send the response to GFM
    lwswitch::fmMessage *pFmMessage = new lwswitch::fmMessage();
    FMIntReturn_t retVal;
    lwswitch::lwlinkResponseMsg *rspMsg = new lwswitch::lwlinkResponseMsg();

    // fill the response msg based on the actual request from GFM
    switch ( getReqType() ) {
        case lwswitch::FM_LWLINK_ENABLE_TX_COMMON_MODE: 
        case lwswitch::FM_LWLINK_DISABLE_TX_COMMON_MODE:
        case lwswitch::FM_LWLINK_CALIBRATE:
        case lwswitch::FM_LWLINK_ENABLE_DATA:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        case lwswitch::FM_LWLINK_OPTICAL_INIT_LINKS:
        case lwswitch::FM_LWLINK_OPTICAL_ENABLE_IOBIST:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_TX:
        case lwswitch::FM_LWLINK_OPTICAL_START_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_CHECK_PRETRAIN_RX:
        case lwswitch::FM_LWLINK_OPTICAL_STOP_PRETRAIN:
        case lwswitch::FM_LWLINK_OPTICAL_DISABLE_IOBIST:
#endif
        case lwswitch::FM_LWLINK_INITPHASE1:
        case lwswitch::FM_LWLINK_RX_INIT_TERM:
        case lwswitch::FM_LWLINK_SET_RX_DETECT:
        case lwswitch::FM_LWLINK_GET_RX_DETECT:
        case lwswitch::FM_LWLINK_INITNEGOTIATE:
        case lwswitch::FM_LWLINK_INITPHASE5:
        case lwswitch::FM_LWLINK_INIT: {
            genGenericNodeIoctlResp( rspMsg );
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
        case lwswitch::FM_LWLINK_SWITCH_TRAINING_FAILED_LINK_INFO: {
            genSwitchTrainingFailedLinkInfoResp( rspMsg );
            break;
        }
        case lwswitch::FM_LWLINK_GET_DEVICE_LWLINK_STATE: {
            genGetDeviceLwlinkStateResp( rspMsg );
            break;
        }
        default:{
            FM_LOG_ERROR( "unknown LWLink initialization request type while preparing response" );
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
        FM_LOG_WARNING( "error while sending LWLink initialization request complete message to fabric manager" );
    }

    // free the allocated  message and return final status
    delete( pFmMessage );

}

void
LocalFMLWLinkReqInit::genGenericNodeIoctlResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // these response types are same as there is no request specific data to 
    // return other than the overall status, which is part of the lwlinkResponseMsg.
    // Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitRspMsg *initRspMsg = new lwswitch::lwlinkNodeInitRspMsg();
    rspMsg->set_allocated_nodeinitrspmsg( initRspMsg );
}

void
LocalFMLWLinkReqInit::genLinkInitStatusResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkNodeInitStatusRspMsg *statusRspMsg = new lwswitch::lwlinkNodeInitStatusRspMsg();
    FMLinkInitStatusInfoList::iterator it;

    // copy the link status information to global FM.
    for ( it = mInitStatusList.begin(); it != mInitStatusList.end(); it++ ) {
        // copy device status information
        FMLinkInitStatusInfo fmLinkStatusInfo = *it;
        lwswitch::lwlinkDeviceLinkInitStatus *devStatusMsg = statusRspMsg->add_initstatus();
        devStatusMsg->set_gpuorswitchid( fmLinkStatusInfo.gpuOrSwitchId );
        // copy per link status information of the device
        for ( int idx = 0; idx < LWLINK_MAX_DEVICE_CONN; idx++ ) {
            lwswitch::lwlinkLinkInitStatus *linkStatus = devStatusMsg->add_linkstatus();
            linkStatus->set_linkindex( fmLinkStatusInfo.initStatus[idx].linkIndex );
            linkStatus->set_status( fmLinkStatusInfo.initStatus[idx].initStatus );
        }
    }

    statusRspMsg->set_nodeid( mLWLinkDevRepo->getLocalNodeId() );
    rspMsg->set_allocated_nodeinitstatusrspmsg( statusRspMsg );
}

void
LocalFMLWLinkReqInit::genResetSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // no request specific data to return other than the overall status, which is part of the
    // parent lwlinkResponseMsg. Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitResetSwitchLinksRspMsg *resetRspMsg = new lwswitch::lwlinkNodeInitResetSwitchLinksRspMsg();
    rspMsg->set_allocated_nodeinitresetswitchlinksrspmsg( resetRspMsg );
}

void
LocalFMLWLinkReqInit::genResetAllSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // no request specific data to return other than the overall status, which is part of the
    // parent lwlinkResponseMsg. Keeping them as independent for future extensions.
    lwswitch::lwlinkNodeInitResetAllSwitchLinksRspMsg *resetRspMsg = new lwswitch::lwlinkNodeInitResetAllSwitchLinksRspMsg();
    rspMsg->set_allocated_nodeinitresetallswitchlinksrspmsg( resetRspMsg );
}

void
LocalFMLWLinkReqInit::genSwitchTrainingFailedLinkInfoResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    // no request specific data to return other than the overall status, which is part of the
    // parent lwlinkResponseMsg. Keeping them as independent for future extensions.
    lwswitch::lwlinkSwitchTrainingFailedRspMsg *trainingFailedRspMsg = new lwswitch::lwlinkSwitchTrainingFailedRspMsg();
    rspMsg->set_allocated_switchtrainingfailedrspmsg( trainingFailedRspMsg );
}

void
LocalFMLWLinkReqInit::genGetDeviceLwlinkStateResp(lwswitch::lwlinkResponseMsg *rspMsg)
{
    lwswitch::lwlinkGetDeviceLwlinkStateRspMsg *getDeviceLwlinkStateRspMsg = new lwswitch::lwlinkGetDeviceLwlinkStateRspMsg();
    FMLWLinkDevInfoList devList = mLWLinkDevRepo->getDeviceList();
    FMLWLinkDevInfoList::iterator it;

    // copy the received LWLink connection information from LWLink Driver 
    // as the response message to GFM.
    for ( it = devList.begin(); it != devList.end(); it++ ) {
        FMLWLinkDevInfo lwlinkDevInfo = *it;
        FMLWLinkStateInfo linkState;

        // copy Active and Faulty LWLink mask information
        for (uint32_t i = 0; i < lwlinkDevInfo.getNumLinks(); i++) {
            // Get linkstate information
            lwlinkDevInfo.getLinkState(i, linkState);

            lwswitch::lwlinkEndPointInfo *lwEndInfo = getDeviceLwlinkStateRspMsg->add_lwendinfo();
            lwswitch::lwlinkStateInfo *stateInfo = new lwswitch::lwlinkStateInfo();

            // Copy NodeId, DeviceId and LinkIndex information
            lwEndInfo->set_nodeid( mLWLinkDevRepo->getLocalNodeId() );
            lwEndInfo->set_gpuorswitchid( lwlinkDevInfo.getDeviceId() );
            lwEndInfo->set_linkindex( (uint64)i );

            // Copy Link State Information
            stateInfo->set_linkmode( linkState.linkMode);
            stateInfo->set_txsublinkmode( linkState.txSubLinkMode );
            stateInfo->set_rxsublinkmode( linkState.rxSubLinkMode );
            lwEndInfo->set_allocated_state( stateInfo );
        }
    }

    rspMsg->set_allocated_getdevicelwlinkstaterspmsg( getDeviceLwlinkStateRspMsg );
}

void
LocalFMLWLinkReqInit::dumpInfo(std::ostream *os)
{
    *os << "\t\t Dumping Link Node Init Req Information" << std::endl;

    // append base request dump information
    LocalFMLWLinkReqBase::dumpInfo( os );
}
