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
#pragma once

#include "LocalFMLWLinkReqBase.h"

/*****************************************************************************/
/* Fabric Manager Link connection training related requests                */
/*****************************************************************************/

/*
 * This class represents LWLink connection training related requests in LFM context.
 * For connection training requests, both endpoints of the connection can be local
 * (single-node connection) or can be in two different nodes (multi-node connection).
 * For single-node connection, the request will be completed from the same context.
 * However, for multi-node connection, the master Node LFM (the node which receive
 * the message from GFM) must to co-ordinate with peer slave LFM. This means, these
 * requests are not immediately completed and have a life cycle. This class 
 * handles both the Master LFM google protobuf messages and slave LFM messages. A
 * node can be Master LFM node and Slave LFM node at the same time for two different
 * connection train requests. But only one Master LFM and Slave LFM for a specific
 * connection train request. This class mainly handles the following connection 
 * train related GPB messages
 * 
 * GPB messages from GFM
 *  FM_MASTER_LWLINK_CONN_SWITCH_OFF
 *  FM_MASTER_LWLINK_CONN_TRAIN_TO_SAFE
 *  FM_MASTER_LWLINK_CONN_TRAIN_TO_HIGH
 *  FM_MASTER_LWLINK_CONN_TRAIN_HIGH_TO_SAFE
 *  FM_MASTER_LWLINK_CONN_TRAIN_SAFE_TO_OFF
 * GPB messages from Master peer LFM
 *  FM_SLAVE_LWLINK_CONN_SWITCH_OFF
 *  FM_SLAVE_LWLINK_CONN_TRAIN_TO_SAFE
 *  FM_SLAVE_LWLINK_CONN_TRAIN_TO_HIGH
 *  FM_SLAVE_LWLINK_CONN_TRAIN_HIGH_TO_SAFE
 *  FM_SLAVE_LWLINK_CONN_TRAIN_SAFE_TO_OFF
 *  FM_LWLINK_TRAIN_RSP_MASTER_SYNC
 * GPB messages from Slave peer LFM
 *  FM_LWLINK_TRAIN_RSP_SLAVE_SYNC
 *  FM_LWLINK_TRAIN_RSP_SLAVE_CONFIRM
 *  FM_LWLINK_TRAIN_RSP_SLAVE_COMPLETE
 *
 * Each of these messages may ilwoke one of the following LWLink driver 
 * ioctl call to train the connection to desired state.
 *  IOCTL_LWLINK_TRAIN_INTRANODE_CONN -- Used to train if the connection is single-node
 *  IOCTL_LWLINK_TRAIN_INTERNODE_CONN_LINK -- Used to train main link for a multi-node connection
 *  IOCTL_LWLINK_TRAIN_INTERNODE_CONN_SUBLINK -- Used to train sub link for a multi-node connection
 */
 
class LocalFMLWLinkReqConnTrain : public LocalFMLWLinkReqConnBase
{
public:

    LocalFMLWLinkReqConnTrain(lwswitch::fmMessage *pFmMessage,
                              FMConnInterface *ctrlConnIntf,
                              LocalFMLWLinkDrvIntf *linkDrvIntf,
                              LocalFMLWLinkDevRepo *linkDevRepo);

    virtual ~LocalFMLWLinkReqConnTrain();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in LocalFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processNewSlaveRequest(lwswitch::fmMessage *pFmMessage);
    virtual bool processRespConfirm(lwswitch::fmMessage *pFmMessage);
    virtual bool processRespComplete(lwswitch::fmMessage *pFmMessage);
    virtual bool processSlaveSync(lwswitch::fmMessage *pFmMessage);
    virtual bool processMasterSync(lwswitch::fmMessage *pFmMessage);
    virtual bool processReqTimeOut();

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    // helper function to handle single-node connection train
    void doNodeConnTrainReq(void);

    // all of these static methods are callback for each link state transition in the 
    // LWLink protocol. These static methods will then call the actual class members. 
    // Static is required as we can't user non-static member functions without the
    // class object.
    static void _masterOffToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                         bool &bReqCompleted);
    static void _masterSafeToHSCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                        bool &bReqCompleted);
    static void _masterToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                     bool &bReqCompleted);
    static void _masterHSToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                        bool &bReqCompleted);
    static void _masterSafeToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                         bool &bReqCompleted);
    static void _slaveOffToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                        bool &bReqCompleted);
    static void _slaveSafeToHSCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                       bool &bReqCompleted);
    static void _slaveToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                    bool &bReqCompleted);
    static void _slaveHSToSafeCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                       bool &bReqCompleted);
    static void _slaveSafeToOffCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                        bool &bReqCompleted);
    static void _ilwalidTrainCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                      bool &bReqCompleted);
    static void _masterSafeToHSNoPeerCallback(LocalFMLWLinkReqConnTrain *fmTrainReq,
                                              bool &bReqCompleted);

    // these non-static class methods will be called by above static methods
    // and do the actual link training operation.
    void masterOffToSafeCallback(bool &bReqCompleted);
    void masterSafeToHSCallback(bool &bReqCompleted);
    void masterToOffCallback(bool &bReqCompleted);
    void masterHSToSafeCallback(bool &bReqCompleted);
    void masterSafeToOffCallback(bool &bReqCompleted);
    void slaveOffToSafeCallback(bool &bReqCompleted);
    void slaveSafeToHSCallback(bool &bReqCompleted);
    void slaveToOffCallback(bool &bReqCompleted);
    void slaveHSToSafeCallback(bool &bReqCompleted);
    void slaveSafeToOffCallback(bool &bReqCompleted);
    void masterSafeToHSNoPeerCallback(bool &bReqCompleted);

    // helper function to ilwoke main-link and sub-link based LWLink
    // driver training ioctls.
    int doSlaveSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                 lwlink_sublink_train_type toLinkState);
    int doMasterSublinkTrainIoctl(lwlink_train_internode_conn_sublink *subLinkParam,
                                  lwlink_sublink_train_type toLinkState);
    int doSlaveMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                  lwlink_link_train_type toLinkState);
    int doMasterMainlinkTrainIoctl(lwlink_train_internode_conn_link *linkParam,
                                   lwlink_link_train_type toLinkState);

    typedef void (*TrainHndlrCallback)(LocalFMLWLinkReqConnTrain* , bool&);

    TrainHndlrCallback mTrainHndlrCB;

    // these state represents a multi-node connection train requests
    // life-cycle in both Master and Slave peer LFM.
    enum LinkTrainReqStates {
        REQ_STATE_TRAIN_NEW_REQUEST = 1,
        REQ_STATE_TRAIN_SLAVE_CONFIRMATION,
        REQ_STATE_TRAIN_SLAVE_SUB_STATE,
        REQ_STATE_TRAIN_MASTER_SUB_STATE,
        REQ_STATE_TRAIN_FINAL_SLAVE_RESP,

    };

    void setLwrrentState(LinkTrainReqStates state) { mReqState = state; }
     LinkTrainReqStates getLwrrentState( ) { return mReqState; }

    LinkTrainReqStates mReqState;
    // For for internode parallel link training, skip sublink training in INITOPTIMIZE_TO_HIGH as it is already done by the INITOPTIMIZE step
    bool mSkipSublinkTraining;
};
