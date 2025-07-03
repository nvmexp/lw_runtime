
#pragma once

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>

#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMLWLinkReqConnTrain.h"


class DcgmFMLWLinkReqBase;
/*****************************************************************************/
/*  Fabric Manager Link Training GPB Message Handler                         */
/*****************************************************************************/

/*
 * This class act as a front-end message handler for all the LWLink GPB messages
 * received by Local Fabric Manager and maintain the life cycle of each requests.
 * Each message handler will create an instance of the corresponding
 * LWLink Request object and pass all the events (like new request, slave sync,
 * timeout etc) to those objects using pre-defined interfaces (virtual functions)
 * This class don't ilwoke any LWLink Driver IOCTLs or implement the actual
 * message handling.
 */

class DcgmFMLWLinkMsgHndlr : public FMMessageHandler
{
public:
    DcgmFMLWLinkMsgHndlr(FMConnInterface *ctrlConnIntf,
                         DcgmFMLWLinkDrvIntf *linkDrvIntf,
                         DcgmLFMLWLinkDevRepo *linkDevRepo);

    virtual ~DcgmFMLWLinkMsgHndlr();

    // implementation of virtual methods for message and event handling

    virtual void handleMessage(lwswitch::fmMessage *pFmMessage);

    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    virtual void dumpInfo(std::ostream *os);

private:

    // Defines the context where each train request events is originated
    enum TrainReqCallCtx {
        MASTER_REQ_RECVD = 1,  // received a request from GFM
        SLAVE_REQ_RECVD,       // received a request from Master LFM (multi-node)
        SLAVE_RESP_CONFIRM,    // received train confirmation from slave FM (multi-node)
        SLAVE_RESP_COMPLETE,   // received a train complete response from slave FM (multi-node)
        SLAVE_RESP_SYNC,       // on master LFM, received a sync response from slave FM
        MASTER_RESP_SYNC,      // on slave LFM, received a sync response from master FM
        REQ_TIMED_OUT,         // the request is timed out, IOCTL is taking more than expected.
    };

    // helper handlers for each GPB messages
    void handleMasterConnTrainReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleSlaveConnTrainReqMsg(lwswitch::fmMessage *pFmMessage);
    void handleSlaveConfirmMsg(lwswitch::fmMessage *pFmMessage);
    void handleInitMsg(lwswitch::fmMessage *pFmMessage);
    void handleDiscoverMsg(lwswitch::fmMessage *pFmMessage);
    void handleConnectionMsg(lwswitch::fmMessage* pFmMessage);
    void handleSlaveCompletemMsg(lwswitch::fmMessage *pFmMessage);
    void handleSlaveSyncMsg(lwswitch::fmMessage *pFmMessage);
    void handleMasterSyncMsg(lwswitch::fmMessage *pFmMessage);

    void deliverMessage(DcgmFMLWLinkReqBase *fmTrainReq,
                        TrainReqCallCtx callCtx,
                        bool &bReqCompleted,
                        lwswitch::fmMessage *pFmMessage);

    // helper functions for managing all the outstanding requests
    void addToReqTrackingTable(DcgmFMLWLinkReqBase *fmTrainReq);
    void removeFromReqTrackingTable(uint64 reqId);
    void removeFromReqTrackingTable(DcgmFMLWLinkReqBase *fmTrainReq);
    DcgmFMLWLinkReqBase* getReqFromTrackingTable(uint64 reqId);

    void handleNodeDisconnect(uint32 nodeId);
    void handleGFMDisconnect();

    typedef std::map <uint64, DcgmFMLWLinkReqBase*> DcgmFMLWLinkReqBaseMap;
    DcgmFMLWLinkReqBaseMap  mFMTrainReqMap;  // outstanding requests

    FMConnInterface *mCtrlConnIntf;
    DcgmFMLWLinkDrvIntf *mLWLinkDrvIntf;
    DcgmLFMLWLinkDevRepo *mLWLinkDevRepo;
};
