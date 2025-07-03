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
#include <g_lwconfig.h>


/*****************************************************************************/
/* Fabric Manager Link connection parallel training related requests                */
/*****************************************************************************/

/*
 * This class represents LWLink connection training related requests in LFM context.
 * For connection parallel training requests, both endpoints of the connection can be local
 * (single-node connection) or can be in two different nodes (multi-node connection).
 * TODO: multi-node connections not handled yet
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
 *  FM_MASTER_LWLINK_CONN_PARALLEL_SWITCH_OFF
 *  FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_SAFE
 *  FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_TO_HIGH
 *  FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_HIGH_TO_SAFE
 *  FM_MASTER_LWLINK_CONN_TRAIN_PARALLEL_SAFE_TO_OFF
 * 
 * Each of these messages may ilwoke one of the following LWLink driver 
 * ioctl call to train the connection to desired state.
 *  IOCTL_LWLINK_TRAIN_PARALLEL_INTRANODE_CONNS -- Used to train if the connection is single-node
 */
typedef struct endPointInfo {
    uint32 nodeId;
    uint64 gpuOrSwitchId;
    uint32 linkIndex;
    lwlink_link_state linkState;
    FMLWLinkQualityInfo qualityInfo;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FMLWLinkGradingValues gradingValues;
    FMLWLinkFomValues fomValues;
#endif
} endPointInfo;

typedef struct linkInfo {
    endPointInfo masterEnd;
    endPointInfo slaveEnd;
} linkInfo;
 
class LocalFMLWLinkReqConnTrainParallel : public LocalFMLWLinkReqBase
{
public:

    LocalFMLWLinkReqConnTrainParallel(lwswitch::fmMessage *pFmMessage,
                                      FMConnInterface *ctrlConnIntf,
                                      LocalFMLWLinkDrvIntf *linkDrvIntf,
                                      LocalFMLWLinkDevRepo *linkDevRepo);

    virtual ~LocalFMLWLinkReqConnTrainParallel();

    // implementation of framework for handling various request types and events.
    // Declared as pure virtual function in LocalFMLWLinkReqBase
    virtual bool processNewMasterRequest(lwswitch::fmMessage *pFmMessage);

    // debug functions
    void dumpInfo(std::ostream *os);

private:

    // helper function to handle single-node connection train
    void doNodeConnTrainParallelReq(lwswitch::fmMessage *pFmMessage);

    void doParallelInitoptimizeReq(lwswitch::fmMessage *pFmMessage);

    void doParallelPostInitoptimizeReq(lwswitch::fmMessage *pFmMessage);

    void doParallelEnableInfModeReq(lwswitch::fmMessage *pFmMessage);

    void doParallelInternodeToOffReq(lwswitch::fmMessage *pFmMessage);

    void doParallelInternodeToHighSpeedReq(lwswitch::fmMessage *pFmMessage);

    void doParallelEnableMaintenanceReq(lwswitch::fmMessage *pFmMessage, bool isTx);

    void doParallelEnableMaintenanceTxReq(lwswitch::fmMessage *pFmMessage);

    void doParallelEnableMaintenanceRxReq(lwswitch::fmMessage *pFmMessage);

    void doParallelDisableInfModeReq(lwswitch::fmMessage *pFmMessage);

    void doParallelEnableForceEqReq(lwswitch::fmMessage *pFmMessage);

    void doParallelDisableForceEqReq(lwswitch::fmMessage *pFmMessage);

    void doParallelCheckEomStatusReq(lwswitch::fmMessage *pFmMessage);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void doParallelGetGradingAndFomValuesReq(lwswitch::fmMessage *pFmMessage);
#endif

    bool getPhyIdFromDeviceId(FMGpuOrSwitchId_t deviceId, FMPhysicalId_t &physicalId);
    void doParallelGetLinkStateReq(lwswitch::fmMessage *pFmMessage);

    void generateTrainParallelRespMsg(lwswitch::fmMessage *pFmMessage,
                                      lwswitch::FabricManagerMessageType msgType,
                                      int respStatus);
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

    virtual void sendRequestCompletion(void);

    std::vector<linkInfo> mLwlinks;

    bool mMasterReq;

    // link state information

    lwswitch::FabricManagerMessageType mSlaveReqType;
};
