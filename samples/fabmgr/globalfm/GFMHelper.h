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

#include <list>

#include "FMCommonTypes.h"
#include "GlobalFMLWLinkIntf.h"
#include "FMTopologyValidator.h"
#include "GlobalFmFabricParser.h"
#include "FMDevInfoMsgHndlr.h"
#include "topology.pb.h"
#include <g_lwconfig.h>

class FMFabricParser;
class GlobalFMLWLinkDevRepo;
class FMLWLinkDetailedConnInfo;

/*****************************************************************************/
/*  Fabric Manager LWLink GFM helper routines                                */
/*****************************************************************************/

/*
 * This class will abstract the helper functions needed by GlobalFabricManager. 
 * The idea is to keep all the functions as static so that no state, object life-time 
 * management and locking for this class.
 */

class GFMHelper
{
public:
    static int lwLinkInitializeAllNodes(GlobalFMLWLinkIntf *linkTrainIntf, 
                                        FMFabricParser* pConfig,
                                        GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkInitializeNode(uint32 nodeId,
                                    GlobalFMLWLinkIntf *linkTrainIntf, 
                                    GlobalFMLWLinkDevRepo &linkDevRepo);
    
    static int lwLinkGetAllNodeLinkInitStatus(GlobalFMLWLinkIntf *linkTrainIntf,
                                              FMFabricParser* pConfig, 
                                              GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkGetAllNodeDeviceLwlinkState(GlobalFMLWLinkIntf *linkTrainIntf,
                                                 FMFabricParser* pConfig,
                                                 GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkDiscoverIntraNodeConnOnNodes(FMFabricParser *pConfig,
                                                  GlobalFMLWLinkIntf *linkTrainIntf,
                                                  GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkDiscoverIntraNodeConnOnNode(uint32 nodeId,
                                                 GlobalFMLWLinkIntf *linkTrainIntf,
                                                 GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkGetIntraNodeConnOnNodes(uint32 nodeId,
                                             GlobalFMLWLinkIntf *linkTrainIntf,
                                             GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkDiscoverInterNodeConnections(GlobalFMLWLinkIntf *linkTrainIntf,
                                                  FMFabricParser *pConfig,
                                                  GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkReadInterNodeLinkSids(GlobalFMLWLinkIntf *linkTrainIntf,
                                           FMFabricParser *pConfig,
                                           GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkTrainIntraNodeConnections(GlobalFabricManager *gfm,
                                               GlobalFMLWLinkIntf *linkTrainIntf,
                                               GlobalFMLWLinkConnRepo &linkConnRepo,
                                               GlobalFMLWLinkDevRepo &linkDevRepo,
                                               FMLWLinkTrainType trainTo);

    static int lwLinkTrainIntraNodeConnectionsParallel(GlobalFabricManager *gfm,
                                                       GlobalFMLWLinkIntf *linkTrainIntf,
                                                       GlobalFMLWLinkConnRepo &linkConnRepo,
                                                       GlobalFMLWLinkDevRepo &linkDevRepo,
                                                       FMLWLinkTrainType trainTo);

    static int lwlinkAddInterNodeConnections(GlobalFMLWLinkIntf *linkTrainIntf,
                                             GlobalFMLWLinkConnRepo &linkConnRepo,
                                             GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkTrainInterNodeConnections(GlobalFabricManager *gfm,
                                               GlobalFMLWLinkIntf *linkTrainIntf,
                                               GlobalFMLWLinkConnRepo &linkConnRepo,
                                               GlobalFMLWLinkDevRepo &linkDevRepo,
                                               FMLWLinkTrainType trainTo);

    static int lwLinkTrainInterNodeConnectionsParallel(GlobalFabricManager *gfm,
                                                       GlobalFMLWLinkIntf *linkTrainIntf,
                                                       GlobalFMLWLinkConnRepo &linkConnRepo,
                                                       GlobalFMLWLinkDevRepo &linkDevRepo);

    static int trainLWLinkConnection(GlobalFabricManager *gfm,
                                     GlobalFMLWLinkIntf *linkTrainIntf,
                                     GlobalFMLWLinkConnRepo &linkConnRepo,
                                     GlobalFMLWLinkDevRepo &linkDevRepo,
                                     FMLWLinkDetailedConnInfo *connInfo,
                                     FMLWLinkTrainType trainTo);

    static int lwLinkSendResetSwitchLinks(uint32 nodeId,
                                          uint64 switchPhysicalId,
                                          uint64 linkMask,
                                          GlobalFMLWLinkIntf *linkTrainIntf);

    static int lwLinkResetAllSwitchLinks(uint32 nodeId, GlobalFMLWLinkIntf *linkTrainIntf);

    static int getLWLinkDeviceInfoFromNode(uint32 nodeId,
                                           GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                           GlobalFMLWLinkDevRepo &linkDevRepo);

    static int getLWLinkDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                               GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                               GlobalFMLWLinkDevRepo &linkDevRepo);

    static int getLWSwitchDeviceInfoFromNode(uint32 nodeId,
                                             GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                             FMLWSwitchInfoMap &lwswitchInfoMap,
                                             FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap);

    static int getLWSwitchDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                                 GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                                 FMLWSwitchInfoMap &lwswitchInfoMap,
                                                 FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap);

    static int getGpuDeviceInfoFromNode(uint32 nodeId,
                                        GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                        FMGpuInfoMap &gpuInfoMap,
                                        FMExcludedGpuInfoMap &excludedGpuInfoMap);

    static int getGpuDeviceInfoFromAllNodes(FMFabricParser *pConfig,
                                            GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                            FMGpuInfoMap &gpuInfoMap,
                                            FMExcludedGpuInfoMap &excludedGpuInfoMap);

    static int getGpuLWLinkSpeedInfoFromAllNodes(FMFabricParser *pConfig,
                                                 GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                                 FMGpuLWLinkSpeedInfoMap &gpuLinkSpeedInfoMap);

    static void updateDeviceAndConnLinkState(GlobalFMLWLinkDevRepo &linkDevRepo,
                                             FMLWLinkDetailedConnInfo *connInfo,
                                             FMLWLinkConnTrainResp &connTrainResp);

    static void updateDeviceAndConnEndPointState(GlobalFMLWLinkDevRepo &linkDevRepo,
                                                 FMLWLinkDetailedConnInfo *connInfo,
                                                 FMLWLinkConnTrainResp &connTrainResp,
                                                 bool isMaster);

    static void getLWLinkConnsRepoByLWLinkDriverId(uint32_t nodeId, uint64_t lwlinkDriverId,
                                                   GlobalFMLWLinkConnRepo &linkConnRepo,
                                                   GlobalFMLWLinkConnRepo &linkConnRepoByDriverId);

    static uint32_t getNumBaseboard(uint32_t nodeId, FMLWSwitchInfoMap &lwswitchInfoMap,
                                    FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap);

    static int lwLinkSendSwitchTrainingFailedLinkInfo(uint32 nodeId,
                                                      uint64 switchPhysicalId,
                                                      uint64 attemptedLinkMask0,
                                                      uint64 failedLinkMask0,
                                                      GlobalFMLWLinkIntf *linkTrainIntf);

    static void logErrAndThrowException(int errorOclwrred, int errCode, const char *errMessage);

    static lwSwitchArchType driverToLwSwitchArchType(uint32_t driverArchType);
    static bool getFMVersionInfoForAllNodes(FMFabricParser *pConfig,
                                            GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr);

    static bool getArchNameForArchType(lwSwitchArchType archType, const char **name);

    static int lwLinkTrainInterNodeConnectionsParallelGetStatus(GlobalFMLWLinkIntf *linkTrainIntf,
                                                                GlobalFMLWLinkConnRepo &linkConnRepo,
                                                                GlobalFMLWLinkDevRepo &linkDevRepo,
                                                                FMFabricParser *pConfig,
                                                                FMLWLinkTrainType trainTo);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    static void getGradingAndFomValues( GlobalFabricManager *gfm,
                                        GlobalFMLWLinkIntf *linkTrainIntf,
                                        GlobalFMLWLinkConnRepo &failedConnections,
                                        GlobalFMLWLinkDevRepo &linkDevRepo );
#endif
private:

    typedef std::list<uint32_t> NodeIDList;

    static int lwLinkTrainInterNodeConnectionsParallelDoStep(GlobalFMLWLinkIntf *linkTrainIntf,
                                                             GlobalFMLWLinkConnRepo &linkConnRepo,
                                                             GlobalFMLWLinkDevRepo &linkDevRepo,
                                                             FMFabricParser *pConfig,
                                                             FMLWLinkTrainType trainTo);

    static void lwLinkTrainInterNodeConnectionsRemoveHighEomLinks(GlobalFMLWLinkConnRepo &linkConnRepo);

    static void sendRequestToTrainLinks(LWLinkInterNodeConns interConnList,
                                        GlobalFMLWLinkIntf *linkTrainIntf,
                                        std::map<uint64, FMLWLinkDetailedConnInfo*> &requestIds, 
                                        std::map<uint64, uint64> &requestIdPairs,
                                        FMLWLinkTrainType trainType);

    static void waitForTrainingLinkReqs(GlobalFabricManager *gfm,
                                        GlobalFMLWLinkIntf *linkTrainIntf,
                                        std::map<uint64, FMLWLinkDetailedConnInfo*> &requestIds, 
                                        std::map<uint64, uint64> &requestIdPairs,
                                        GlobalFMLWLinkDevRepo &linkDevRepo,
                                        FMLWLinkTrainType trainTo);
	
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    static int lwLinkTrainInterNodeConnectionsParallelDoForceEQ(GlobalFabricManager *gfm,
                                                                GlobalFMLWLinkIntf *linkTrainIntf,
                                                                GlobalFMLWLinkConnRepo &linkConnRepo,
                                                                GlobalFMLWLinkDevRepo &linkDevRepo);
    
    static void updateConnEndPointFomValues(FMLWLinkDetailedConnInfo *connInfo,
                                            FMLWLinkFomValues &fomValues,
                                            bool isMaster);

    static void updateConnEndPointGradingValues(FMLWLinkDetailedConnInfo *connInfo,
                                                FMLWLinkGradingValues &gradingValues,
                                                bool isMaster);

#endif

    static void updateConnEndPointLinkQualityInfo(FMLWLinkDetailedConnInfo *connInfo,
                                                  FMLWLinkQualityInfo &linkQualityInfo,
                                                  bool isMaster);

    static int lwLinkInitializeNodes(NodeIDList &nodeIdList,
                                     GlobalFMLWLinkIntf *linkTrainIntf,
                                     GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkInitDoStepForNodes(NodeIDList &nodeIdList,
                                        GlobalFMLWLinkIntf *linkTrainIntf,
                                        lwswitch::FabricManagerMessageType msgType);

    static int lwLinkTrainWaitForReponsesParallel(GlobalFabricManager *gfm,
                                                  GlobalFMLWLinkIntf *linkTrainIntf,
                                                  GlobalFMLWLinkDevRepo &linkDevRepo,
                                                  FMLWLinkTrainType trainTo,
                                                  std::map<uint64, FMLWLinkDetailedConnInfoList*> &requestIds);

    static int lwLinkTrainWaitForReponsesParallelInternode(GlobalFMLWLinkIntf *linkTrainIntf,
                                                           GlobalFMLWLinkDevRepo &linkDevRepo,
                                                           FMLWLinkTrainType trainTo,
                                                           std::map<uint64, FMLWLinkDetailedConnInfoList*> &requestIds);


    static int lwLinkGetLinkInitStatusForNodes(NodeIDList &nodeIdList,
                                               GlobalFMLWLinkIntf *linkTrainIntf,
                                               GlobalFMLWLinkDevRepo &linkDevRepo);

    static int lwLinkGetIntraNodeConns(NodeIDList &nodeIdList,
                                       GlobalFMLWLinkIntf *linkTrainIntf,
                                       GlobalFMLWLinkConnRepo &linkConnRepo);

    static int lwLinkSendDiscoverConnOnNodes(NodeIDList &nodeIdList,
                                             GlobalFMLWLinkIntf *linkTrainIntf);

    static int lwLinkWriteDiscoveryToken(uint32 nodeId,
                                         GlobalFMLWLinkIntf *linkTrainIntf,
                                         FMLWLinkWriteDiscoveryTokenResp &writeTokenResp);

    static int lwLinkReadLinkSids(GlobalFMLWLinkIntf *linkTrainIntf,
                                  FMFabricParser *pConfig,
                                  FMLWLinkSidList &interNodeLinkSidList,
                                  std::map<uint64, uint32> &sidToNodeIdMap,
                                  std::map<uint64, uint64> &sidToGpuOrSwitchIdMap);

    static int lwLinkReadDiscoveryToken(uint32 lwrWriteTokenNodeId,
                                        GlobalFMLWLinkIntf *linkTrainIntf,
                                        FMFabricParser *pConfig,
                                        std::map<uint32, FMLWLinkReadDiscoveryTokenResp> &readTokenResps);

    static void lwLinkCorrelateConnections(FMLWLinkWriteDiscoveryTokenResp &writeTokenResp,
                                           std::map<uint32, FMLWLinkReadDiscoveryTokenResp> &readTokenResps,
                                           GlobalFMLWLinkConnRepo &linkConnRepo);

    static void lwLinkCorrelateLinkSids(FMLWLinkSidList &interNodeLinkSidList,
                                        std::map<uint64, uint32> &sidToNodeIdMap,
                                        std::map<uint64, uint64> &sidToGpuOrSwitchIdMap,
                                        GlobalFMLWLinkConnRepo &linkConnRepo);

    static bool lwLinkIsInterNodeConnectionExists(GlobalFMLWLinkConnRepo &linkConnRepo,
                                                  FMLWLinkConnInfo &newConn);

    static int waitForLinkRequestToComplete(GlobalFMLWLinkIntf *linkTrainIntf,
                                            std::list<uint64> requestIds,
                                            std::string errorCtx);

    static void genAddInterNodeConnLinkReqMsg(GlobalFMLWLinkDevRepo &linkDevRepo,
                                              FMLWLinkReq &linkReq,
                                              FMLWLinkEndPointInfo &localEndInfo,
                                              FMLWLinkEndPointInfo &remoteEndInfo);

    static void copyLWSwitchDeviceInfo(DevInfoReqResult &reqResult,
                                       FMLWSwitchInfoMap &lwswitchInfoMap,
                                       FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap);

    static void copyGpuDeviceInfo(DevInfoReqResult &reqResult,
                                  FMGpuInfoMap &gpuInfoMap,
                                  FMExcludedGpuInfoMap &excludedGpuInfoMap);

    static void copyGpuLWLinkSpeedInfo(DevInfoReqResult &reqResult,
                                       FMGpuLWLinkSpeedInfoMap &gpuLinkSpeedInfoMap);

    static const char* getLWLinkTrainTypeString(FMLWLinkTrainType trainType);

    static const char* getLWLinkInitTypeString(lwswitch::FabricManagerMessageType);

    static int getLWSwitchDeviceInfoFromNodes(NodeIDList &nodeIdList,
                                              GlobalFMDevInfoMsgHdlr *devInfoMsgHdlr,
                                              FMLWSwitchInfoMap &lwswitchInfoMap,
                                              FMExcludedLWSwitchInfoMap &excludedLwswitchInfoMap);

    static uint32_t getBaseboardSlotNumberFromSwitchPhysicaId(uint32_t physicalId);
    static void getTrainTypeVector(FMLWLinkTrainType trainType, vector<FMLWLinkTrainType> &trainTypes);
    static int lwLinkGetDeviceLwlinkStateForNodes(NodeIDList &nodeIdList,
                                                  GlobalFMLWLinkIntf *linkTrainIntf,
                                                  GlobalFMLWLinkDevRepo &linkDevRepo);

    static void updateDeviceLinkState(GlobalFMLWLinkDevRepo &linkDevRepo,
                                      FMLWLinkGetDeviceLwlinkStateResp &deviceLwlinkStateList);

};
