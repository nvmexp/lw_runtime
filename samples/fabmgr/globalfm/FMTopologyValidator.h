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

/*****************************************************************************/
/*  Abstract all the runtime validation of topology file                     */
/*****************************************************************************/

/*
 * This class provides the required interfaces to Global Fabric Manager to
 * validate the topology file content againist what each Local Fabric Manager
 * reported. This includes the number of expected LWSwitches, GPUs, LWLink
 * connections etc.
 */

#include "FMCommonTypes.h"
#include "GlobalFabricManager.h"
#include "GlobalFmDegradedModeMgr.h"
#include "GlobalFmFabricParser.h"
#include "FMLWLinkDeviceRepo.h"

class FMFabricParser;
class GlobalFabricManager;

class FMTopologyValidator
{
    friend class GlobalFmDegradedModeMgr;

public:
    FMTopologyValidator(GlobalFabricManager *pGfm);

    ~FMTopologyValidator();

    // validation routies
    int validateTopologyInfo(void);

    bool mapGpuIndexByLWLinkConns(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);

    int disableGpus(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);

    int disableSwitches(void);

    bool isAllIntraNodeTrunkConnsActive(void);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // returns a list of all failed ports on all failed nodes
    bool isTrunkConnsActive(GlobalFMLWLinkConnRepo &failedConnections, int &numConfigTrunks); 

    // returns the number of internode trunk connections that are in Active state
    int numActiveInternodeTrunks();
    int numActiveIntranodeTrunks();

    void addFailedConnection(GlobalFMLWLinkConnRepo &failedConnections, TopologyLWLinkConn configConn);

    bool isNodeConfigured(FMNodeId_t nodeId);

#endif
    void checkLWLinkInitStatusForAllDevices();

    bool isSwitchPortConnected(uint32_t nodeId, uint32_t physicalId, uint32_t linkIndex);

    uint32_t getIntraNodeTrunkConnCount(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo);
    uint32_t getAccessConnCount(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo);
    bool isIntraNodeTrunkConnsActive(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                                     GlobalFMLWLinkDevRepo &lwLinkDevRepo);
    bool isAccessConnsActive(uint32_t nodeId, GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                             GlobalFMLWLinkDevRepo &lwLinkDevRepo);
    void logNonDetectedLWLinkConns(std::list<uint32> &missingConnLinkIndex, uint32_t nodeId,
                                   uint32_t physicalId, uint64 deviceType);

private:

    typedef struct {
        uint32_t nodeId;
        uint32_t physicalId;
        uint32_t enumIndex;
        uint32 linkIndex;
    } FMDetailedEndPointInfo;

    // Check if the configured connection was discovered at run time
    bool checkConnInDiscoveredConnections( TopologyLWLinkConn &configConn );

    int validateNodeTopologyInfo(NodeConfig *pNodeCfg, uint32_t switchCount,
                                 uint32_t lwLinkIntraConnCount, uint32_t lwLinkInterConnCount,
                                 uint32_t gpuCount, uint32_t excludedGpuCount);

    bool mapGpuIndexForOneNode(uint32_t nodeId,
                               FMGpuLWLinkConnMatrixList &connMatrix);

    bool updateGpuConnectionMatrixInfo(FMGpuLWLinkConnMatrixList &connMatrixList,
                                       FMDetailedEndPointInfo gpuEndPointInfo,
                                       TopologyLWLinkConnEndPoint topoGpuEndPoint,
                                       FMUuid_t &uuid);

    bool getTopologyLWLinkConnGpuEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                          TopologyLWLinkConnEndPoint &gpuEndPoint);

    bool getTopologyLWLinkConnBySwitchEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                               TopologyLWLinkConn &topoLWLinkConn);

    bool getTopologyLWLinkConnByGpuEndPoint(TopologyLWLinkConnEndPoint &gpuEndPoint,
                                            TopologyLWLinkConn &topoLWLinkConn);

    bool isLWLinkTrunkConnection(FMLWLinkDetailedConnInfo *lwLinkConn);

    bool getAccessConnDetailedEndPointInfo(FMLWLinkDetailedConnInfo *lwLinkConn,
                                           FMDetailedEndPointInfo &switchEndPointInfo,
                                           FMDetailedEndPointInfo &gpuEndPointInfo,
                                           FMLWSwitchInfo &switchInfo,
                                           FMGpuInfo_t &gpuInfo);

    bool verifyAccessConnection(FMLWLinkDetailedConnInfo *lwLinkConn);

    bool getLWSwitchInfoByLWLinkDevInfo(uint32 nodeId,
                                        FMLWLinkDevInfo &lwLinkDevInfo,
                                        FMLWSwitchInfo &lwSwitchInfo);

    bool getGpuInfoByLWLinkDevInfo(uint32 nodeId,
                                   FMLWLinkDevInfo &lwLinkDevInfo,
                                   FMGpuInfo_t &gpuInfo);

    bool getAccessConnGpuLinkMaskInfo(FMLWLinkDetailedConnInfo *lwLinkConn,
                                      uint64 &gpuEnabledLinkMask);

    int getGpuConnNum(uint32 nodeId, uint32_t physicalIdx);
    int getMaxGpuConnNum(void);

    bool compareLinks(FMLWLinkDetailedConnInfo *discoveredConn, TopologyLWLinkConn configConn);

    void checkLWSwitchLinkInitStatus(void);
    void checkLWSwitchDeviceLinkInitStatus(uint32_t nodeId,
                                           FMLWLinkDevInfo &devInfo);
    void checkGpuLinkInitStatus(void);
    void checkGpuDeviceLinkInitStatus(uint32_t nodeId,
                                      FMLWLinkDevInfo &devInfo);

    uint32_t getInterNodeTrunkConnCount(uint32_t nodeId);

    bool isGpuConnectedToSwitch(GpuKeyType key);
    void getUnmappedGpusForOneNode(uint32_t nodeId, FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap,
                                   FMGpuInfoList &gpuList);

    void checkForNonDetectedLWLinkConns(int nodeId);

    int disableGpusForNonDegradedMode(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);
    int disableGpusForDegradedMode(FMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);

    GlobalFabricManager *mGfm;
};
