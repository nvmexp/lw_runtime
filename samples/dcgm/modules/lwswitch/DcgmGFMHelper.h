#pragma once

#include <list>

#include "DcgmFMCommon.h"
#include "DcgmFMLWLinkIntf.h"
#include "DcgmFMLWLinkConnRepo.h"
#include "DcgmFMTopologyValidator.h"
#include "DcgmFabricParser.h"
#include "DcgmFMDevInfoMsgHndlr.h"
#include <g_lwconfig.h>


class DcgmFabricParser;
class DcgmGFMLWLinkDevRepo;
class DcgmFMLWLinkDetailedConnInfo;

/*****************************************************************************/
/*  Fabric Manager LWLink GFM helper routines                                */
/*****************************************************************************/

/*
 * This class will abstract the helper functions needed by DcgmGlobalFabricManager. 
 * The idea is to keep all the functions as static so that no state, object life-time 
 * management and locking for this class.
 */

class DcgmGFMHelper
{
public:
    static int lwLinkInitializeAllNodes(DcgmFMLWLinkIntf *linkTrainIntf, 
                                        DcgmFabricParser* pConfig,
                                        DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkInitializeNode(uint32 nodeId,
                                    DcgmFMLWLinkIntf *linkTrainIntf, 
                                    DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkDiscoverIntraNodeConnOnNodes(DcgmFabricParser *pConfig,
                                                  DcgmFMLWLinkIntf *linkTrainIntf,
                                                  DcgmFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkDiscoverIntraNodeConnOnNode(uint32 nodeId,
                                                 DcgmFMLWLinkIntf *linkTrainIntf,
                                                 DcgmFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkGetIntraNodeConnOnNodes(uint32 nodeId,
                                             DcgmFMLWLinkIntf *linkTrainIntf,
                                             DcgmFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkDiscoverInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                                  DcgmFabricParser *pConfig,
                                                  DcgmFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkTrainIntraNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmFMLWLinkConnRepo &linkConnRepo,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo,
                                               DcgmLWLinkTrainType trainTo,
                                               bool inErrHdlr = false);

    static int lwlinkAddInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                             DcgmFMLWLinkConnRepo &linkConnRepo,
                                             DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkTrainInterNodeConnections(DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmFMLWLinkConnRepo &linkConnRepo,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo,
                                               DcgmLWLinkTrainType trainTo);

    static int trainLWLinkConnection(DcgmFMLWLinkIntf *linkTrainIntf,
                                     DcgmFMLWLinkConnRepo &linkConnRepo,
                                     DcgmGFMLWLinkDevRepo &linkDevRepo,
                                     DcgmFMLWLinkDetailedConnInfo *connInfo,
                                     DcgmLWLinkTrainType trainTo);

    static int lwLinkSendResetSwitchLinks(uint32 nodeId,
                                          uint64 switchPhysicalId,
                                          uint64 linkMask,
                                          DcgmFMLWLinkIntf *linkTrainIntf,
                                          bool inErrHdlr);

    static int lwLinkResetAllSwitchLinks(uint32 nodeId, DcgmFMLWLinkIntf *linkTrainIntf);

    static int getLWLinkDeviceInfoFromNode(uint32 nodeId,
                                           GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                           DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int getLWLinkDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                               GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo);


    static int getLWSwitchDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                                 GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                                 DcgmFMLWSwitchInfoMap &lwswitchInfoMap);

    static int getGpuDeviceInfoFromNode(uint32 nodeId,
                                        GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                        DcgmFMGpuInfoMap &gpuInfoMap,
                                        DcgmFMGpuInfoMap &blacklistGpuInfoMap);

    static int getGpuDeviceInfoFromAllNodes(DcgmFabricParser *pConfig,
                                            GFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                            DcgmFMGpuInfoMap &gpuInfoMap,
                                            DcgmFMGpuInfoMap &blacklistGpuInfoMap);

private:

    typedef std::list<uint32_t> NodeIDList;

    static int lwLinkInitializeNodes(NodeIDList &nodeIdList,
                                     DcgmFMLWLinkIntf *linkTrainIntf,
                                     DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkEnableCommonModeForNodes(NodeIDList &nodeIdList,
                                              DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkCalibrateNodes(NodeIDList &nodeIdList,
                                    DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkDisableCommonModeForNodes(NodeIDList &nodeIdList,
                                               DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkEnableDataForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    static int lwLinkInitphase1ForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkRxInitTermForNodes(NodeIDList &nodeIdList,
                                        DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkSetRxDetectForNodes(NodeIDList &nodeIdList,
                                         DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkGetRxDetectForNodes(NodeIDList &nodeIdList,
                                         DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkInitnegotiateForNodes(NodeIDList &nodeIdList,
                                           DcgmFMLWLinkIntf *linkTrainIntf);
#endif

    static int lwLinkInitLinkForNodes(NodeIDList &nodeIdList,
                                      DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkGetLinkInitStatusForNodes(NodeIDList &nodeIdList,
                                               DcgmFMLWLinkIntf *linkTrainIntf,
                                               DcgmGFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkGetIntraNodeConns(NodeIDList &nodeIdList,
                                       DcgmFMLWLinkIntf *linkTrainIntf,
                                       DcgmFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkSendDiscoverConnOnNodes(NodeIDList &nodeIdList,
                                             DcgmFMLWLinkIntf *linkTrainIntf);

    static int lwLinkWriteDiscoveryToken(uint32 nodeId,
                                         DcgmFMLWLinkIntf *linkTrainIntf,
                                         DcgmLWLinkWriteDiscoveryTokenResp &writeTokenResp);

    static int lwLinkReadDiscoveryToken(uint32 lwrWriteTokenNodeId,
                                        DcgmFMLWLinkIntf *linkTrainIntf,
                                        DcgmFabricParser *pConfig,
                                        std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp> &readTokenResps);

    static int lwLinkCorrelateConnections(DcgmLWLinkWriteDiscoveryTokenResp &writeTokenResp,
                                          std::map<uint32, DcgmLWLinkReadDiscoveryTokenResp> &readTokenResps,
                                          DcgmFMLWLinkConnRepo &linkConnRepo);

    static bool lwLinkIsInterNodeConnectionExists(DcgmFMLWLinkConnRepo &linkConnRepo,
                                                  DcgmLWLinkConnInfo &newConn);

    static int waitForLinkRequestToComplete(DcgmFMLWLinkIntf *linkTrainIntf,
                                            std::list<uint64> requestIds,
                                            std::string errorCtx);

    static void genAddInterNodeConnLinkReqMsg(DcgmGFMLWLinkDevRepo &linkDevRepo,
                                              DcgmLWLinkReq &linkReq,
                                              DcgmLWLinkEndPointInfo &localEndInfo,
                                              DcgmLWLinkEndPointInfo &remoteEndInfo);

    static void copyLWSwitchDeviceInfo(DevInfoReqResult &reqResult,
                                       DcgmFMLWSwitchInfoMap &lwswitchInfoMap);

    static void copyGpuDeviceInfo(DevInfoReqResult &reqResult,
                                  DcgmFMGpuInfoMap &gpuInfoMap,
                                  DcgmFMGpuInfoMap &blacklistGpuInfoMap);

    static void updateDeviceAndConnLinkState(DcgmGFMLWLinkDevRepo &linkDevRepo,
                                             DcgmFMLWLinkDetailedConnInfo *connInfo,
                                             DcgmLWLinkConnTrainResp &connTrainResp);

    static const char* getLWLinkTrainTypeString(DcgmLWLinkTrainType trainType);
};
