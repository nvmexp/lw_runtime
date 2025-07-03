
#pragma once

#include <iostream>
#include <fstream>
#include <string.h>

#include "dcgm_structs.h"
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMLWLinkDrvIntf.h"
#include "DcgmFMLWLinkDeviceRepo.h"

/*****************************************************************************/
/*  Fabric Manager Link Training request abstraction                         */
/*****************************************************************************/

/*
 * DcgmFMLWLinkReqBase - This class represents an LWLink request in LFM context, 
 * which is received from either GFM or Peer LFM (as part of multi-node) as a 
 * google protobuf message. Also the multi-node connection train LWLink requests 
 * have a life time and will not complete in single call context as it needs to
 * sync with peer LFM. Also each LWLink request have a common action pattern like
 * a new master/slave request is received, a slave/master sync is received, 
 * the request is timed out etc. All of these actions are abstracted so that the
 * upper message handler can operate on these events without worrying about what
 * the actual request is.
 *
 * DcgmFMLWLinkReqConnBase - This class again abstract the connection train request
 * as it is a special case request where we have Master/Slave FM sync.
 *
 * Dependency:
 *  FMConnInterface : Provides methods for sending messages to peer LFM and GFM.
 *  DcgmFMLWLinkDrvIntf : Provides methods for accessing LWLink driver (ioctl)
 *  DcgmLFMLWLinkDevRepo : Provides methods to get PCI, Deviceid etc about each LWLink
 *                         device and a list of LWLink devices as few IOCTLs are per
 *                         device and we need to repeat it for all the devices.
 */

class DcgmFMLWLinkReqBase
{
public:

    DcgmFMLWLinkReqBase(lwswitch::fmMessage *pFmMessage,
                        FMConnInterface *ctrlConnIntf,
                        DcgmFMLWLinkDrvIntf *linkDrvIntf,
                        DcgmLFMLWLinkDevRepo *linkDevRepo);

    virtual ~DcgmFMLWLinkReqBase();

    // framework for handling various request types and events.
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processNewSlaveRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processRespConfirm(lwswitch::fmMessage *pFmMessage);
    virtual bool processRespComplete(lwswitch::fmMessage *pFmMessage);
    virtual bool processSlaveSync(lwswitch::fmMessage *pFmMessage);
    virtual bool processMasterSync(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // basic request related information retrieval function
    uint64 getTrainReqId() { return mTrainReqId; }
    uint32 getDcgmMsgReqId() { return mDcmgMsgReqId; }

    // return true if the request is from a master FM
    // default implementation. Derived classes can override it
    virtual bool isMasterReq() { return false; } 

    // debug functions    
    virtual void dumpInfo(std::ostream *os);

private:

    bool defaultReqHandler(lwswitch::fmMessage *pFmMessage,
                           std::string errorPrefix);
protected:

    // provides interface for sending the request completion
    // depending on context, completion may be to GFM or to peer LFM (slave)
    virtual void sendRequestCompletion(void) = 0;

    lwswitch::FabricManagerMessageType getReqType() { return mReqType; }

    // set/get for the final status of the request (on of DcgmLWLinkErrorCodes value)
    void setCompletionStatus(int status) { mCompStatus = status; }
    int getCompletionStatus() { return mCompStatus; }

    // since the same request can be referenced from multiple context
    // like from the socket thread context when the message is recevied
    // or from a timer context when the message is timedout etc
    void lock(void);
    void unLock(void);

    uint64 mTrainReqId;
    uint32 mDcmgMsgReqId;

    lwswitch::FabricManagerMessageType mReqType;
    int mCompStatus;

    LWOSCriticalSection mLock;
    FMConnInterface *mCtrlConnIntf;
    DcgmFMLWLinkDrvIntf *mLWLinkDrvIntf;
    DcgmLFMLWLinkDevRepo *mLWLinkDevRepo;

};

class DcgmFMLWLinkReqConnBase: public DcgmFMLWLinkReqBase
{
public:

    DcgmFMLWLinkReqConnBase(lwswitch::fmMessage *pFmMessage,
                            FMConnInterface *ctrlConnIntf,
                            DcgmFMLWLinkDrvIntf *linkDrvIntf,
                            DcgmLFMLWLinkDevRepo *linkDevRepo);

    virtual ~DcgmFMLWLinkReqConnBase();

    // connection train can be master/slave type in
    // multi-node training.
    virtual bool isMasterReq() { return mMasterReq; }

    // debug functions    
    virtual void dumpInfo(std::ostream *os);

    // endpoint related information of a connection train request
    uint32 getMasterNodeId() { return mMasterNodeId; }
    uint64 getMasterGpuId() { return mMasterGpuOrSwitchId; }
    uint32 getMasterLinkIndex() { return mMasterLinkIndex; }
    uint32 getSlaveNodeId() { return mSlaveNodeId; }
    uint64 getSlaveGpuId() { return mSlaveGpuOrSwitchId; }
    uint32 getSlaveLinkIndex() { return mSlaveLinkIndex; }

protected:

    lwswitch::FabricManagerMessageType getSlaveReqType() { return mSlaveReqType; }

    // set/get for the link state information returned by LWLink Driver.
    // mainly link, rx, and tx state per endpoint.
    uint32 getMasterLinkMode() { return mMasterLinkState.linkMode; }
    uint32 getMasterTxSubLinkMode() { return mMasterLinkState.txSubLinkMode; }
    uint32 getMasterRxSubLinkMode() { return mMasterLinkState.rxSubLinkMode; }
    uint32 getSlaveLinkMode() { return mSlaveLinkState.linkMode; }
    uint32 getSlaveTxSubLinkMode() { return mSlaveLinkState.txSubLinkMode; }
    uint32 getSlaveRxSubLinkMode() { return mSlaveLinkState.rxSubLinkMode; }
    void setMasterLinkState (lwlink_link_state &linkState) { mMasterLinkState = linkState; }
    void setSlaveLinkState (lwlink_link_state &linkState) { mSlaveLinkState = linkState; }

    // train request completion related protobuf generation
    // the request completion may be to GFM (from master node)
    // or to a peer LFM (from slave FM node)
    virtual void sendRequestCompletion(void);

    void generateTrainRespMsg(lwswitch::fmMessage *pFmMessage,
                              lwswitch::FabricManagerMessageType msgType,
                              int respStatus);

    void sendSlaveConfirmation(void);
    void sendSlaveSyncMsg(void);
    void sendMasterSyncMsg(void);
    dcgmReturn_t sendSlaveTrainReqMsg(lwswitch::FabricManagerMessageType msgType);

    uint32 mMasterNodeId;
    uint64 mMasterGpuOrSwitchId;
    uint32 mMasterLinkIndex;
    uint32 mSlaveNodeId;
    uint64 mSlaveGpuOrSwitchId;
    uint32 mSlaveLinkIndex;
    bool mMasterReq;

    // link state information
    lwlink_link_state mMasterLinkState;
    lwlink_link_state mSlaveLinkState;

    lwswitch::FabricManagerMessageType mSlaveReqType;
};
