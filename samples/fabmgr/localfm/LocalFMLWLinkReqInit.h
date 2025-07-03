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
#pragma once

#include "LocalFMLWLinkReqBase.h"
#include <g_lwconfig.h>

/*****************************************************************************/
/*  Fabric Manager Link initialization request                               */
/*****************************************************************************/

/*
 * This class represents an LWLink link initialization request in LFM context.
 * All the link initialization requests are specific to a node and there is 
 * no master/slave peer LFM sync. This class mainly handles the following
 * initialization related GPB messages from GFM and call the specified 
 * corresponding LWLink driver ioctl. The final status is then returned to GFM.
 *
 * FM_LWLINK_ENABLE_TX_COMMON_MODE   ==> IOCTL_LWLINK_SET_TX_COMMON_MODE
 * FM_LWLINK_DISABLE_TX_COMMON_MODE  ==> IOCTL_LWLINK_SET_TX_COMMON_MODE
 * FM_LWLINK_CALIBRATE               ==> IOCTL_LWLINK_CALIBRATE
 * FM_LWLINK_ENABLE_DATA             ==> IOCTL_LWLINK_ENABLE_DATA
 * FM_LWLINK_INIT                    ==> IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC
 * FM_LWLINK_INIT_STATUS             ==> IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS
 * FM_LWLINK_RESET_SWITCH_LINKS      ==> IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS
 * FM_LWLINK_RESET_ALL_SWITCH_LINKS  ==> IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS
 * FM_LWLINK_GET_DEVICE_LWLINK_STATE ==> IOCTL_LWLINK_GET_DEVICE_LINKS_STATE
 *
 */

class LocalFMLWLinkReqInit : public LocalFMLWLinkReqBase
{
public:

    LocalFMLWLinkReqInit(lwswitch::fmMessage *pFmMessage,
                         FMConnInterface *ctrlConnIntf,
                         LocalFMLWLinkDrvIntf *linkDrvIntf,
                         LocalFMLWLinkDevRepo *linkDevRepo);

    virtual ~LocalFMLWLinkReqInit();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in LocalFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    virtual void sendRequestCompletion(void);

    // helper functions to ilwoke corresponding ioctl calls
    void doEnableTxCommonMode(void);
    void doDisableTxCommonMode(void);
    void doCalibrate(void);
    void doEnableData(void);
    void doInitphase5(void);
    void doLinkInit(void);
    void doLinkInitStatus(void);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void doOpticalInitLinks(void);
    void doOpticalEnableIobist(void);
    void doOpticalStartPretrain(bool isTx);
    void doOpticalCheckPretrain(bool isTx);
    void doOpticalStopPretrain(void);
    void doOpticalDisableIobist(void);
#endif
    void doInitphase1(void);
    void doRxInitTerm(void);
    void doSetRxDetect(void);
    void doGetRxDetect(void);
    void doInitnegotiate(void);

    void doResetSwitchLinks(const lwswitch::lwlinkMsg &linkMsg);
    void doResetAllSwitchLinks(const lwswitch::lwlinkMsg &linkMsg);
    void doSwitchTrainingFailedLinkInfo(const lwswitch::lwlinkMsg &linkMsg);
    void doGetDeviceLwlinkState(void);

    // helper functions to generate response GPB message to GFM
    void genGenericNodeIoctlResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genLinkInitStatusResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genResetSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genResetAllSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genSwitchTrainingFailedLinkInfoResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genGetDeviceLwlinkStateResp(lwswitch::lwlinkResponseMsg *rspMsg);

    FMLinkInitStatusInfoList mInitStatusList;
};
