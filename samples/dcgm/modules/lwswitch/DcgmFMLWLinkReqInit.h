
#pragma once

#include "DcgmFMLWLinkReqBase.h"
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
 * FM_LWLINK_ENABLE_TX_COMMON_MODE  ==> IOCTL_LWLINK_SET_TX_COMMON_MODE
 * FM_LWLINK_DISABLE_TX_COMMON_MODE ==> IOCTL_LWLINK_SET_TX_COMMON_MODE
 * FM_LWLINK_CALIBRATE              ==> IOCTL_LWLINK_CALIBRATE
 * FM_LWLINK_ENABLE_DATA            ==> IOCTL_LWLINK_ENABLE_DATA
 * FM_LWLINK_INIT                   ==> IOCTL_CTRL_LWLINK_LINK_INIT_ASYNC
 * FM_LWLINK_INIT_STATUS            ==> IOCTL_CTRL_LWLINK_DEVICE_LINK_INIT_STATUS
 * FM_LWLINK_RESET_SWITCH_LINKS     ==> IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS
 * FM_LWLINK_RESET_ALL_SWITCH_LINKS ==> IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS
 *
 */

class DcgmFMLWLinkReqInit : public DcgmFMLWLinkReqBase
{
public:

    DcgmFMLWLinkReqInit(lwswitch::fmMessage *pFmMessage,
                        FMConnInterface *ctrlConnIntf,
                        DcgmFMLWLinkDrvIntf *linkDrvIntf,
                        DcgmLFMLWLinkDevRepo *linkDevRepo);

    virtual ~DcgmFMLWLinkReqInit();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in DcgmFMLWLinkReqBase
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
    void doLinkInit(void);
    void doLinkInitStatus(void);
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void doInitphase1(void);
    void doRxInitTerm(void);
    void doSetRxDetect(void);
    void doGetRxDetect(void);
    void doInitnegotiate(void);
#endif
    void doResetSwitchLinks(const lwswitch::lwlinkMsg &linkMsg);
    void doResetAllSwitchLinks(const lwswitch::lwlinkMsg &linkMsg);

    // helper functions to generate response GPB message to GFM
    void genEnableCommonModeResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genDisableCommonModeResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genCalibrateResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genEnableDataResp(lwswitch::lwlinkResponseMsg *rspMsg);
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void genInitphase1Resp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genRxInitTermResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genSetRxDetectResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genGetRxDetectResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genInitnegotiateResp(lwswitch::lwlinkResponseMsg *rspMsg);
#endif
    void genLinkInitResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genLinkInitStatusResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genResetSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg);
    void genResetAllSwitchLinksResp(lwswitch::lwlinkResponseMsg *rspMsg);

    DcgmLinkInitStatusInfoList mInitStatusList;
};
