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

#include "DcgmFMCommon.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFabricParser.h"
#include "DcgmFMLWLinkDeviceRepo.h"

class DcgmFabricParser;
class DcgmGlobalFabricManager;

class DcgmFMTopologyValidator
{
public:
    DcgmFMTopologyValidator(DcgmGlobalFabricManager *pGfm);

    ~DcgmFMTopologyValidator();

    // validation routies
    int validateTopologyInfo(void);

    bool mapGpuIndexByLWLinkConns(DcgmFMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);

    int disableGpus(DcgmFMGpuLWLinkConnMatrixMap &gpuConnMatrixMap);

    int disableSwitches(void);

    bool isAllLWLinkTrunkConnsActive(void);

    void checkLWLinkInitStatusForAllDevices(void);

    bool isSwitchPortConnected(uint32_t nodeId, uint32_t physicalId, uint32_t linkIndex);

    uint32_t getIntraNodeTrunkConnCount(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo);
    uint32_t getAccessConnCount(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo);
    bool isIntraNodeTrunkConnsActive(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo);
    bool isAccessConnsActive(uint32_t nodeId, DcgmFMLWLinkConnRepo &lwLinkConnRepo);

private:

    typedef struct {
        uint32_t nodeId;
        uint32_t physicalId;
        uint32_t enumIndex;
        uint32 linkIndex;
    } DcgmDetailedEndPointInfo;

    int validateNodeTopologyInfo(NodeConfig *pNodeCfg, uint32_t switchCount,
                                 uint32_t lwLinkIntraConnCount, uint32_t lwLinkInterConnCount,
                                 uint32_t gpuCount, uint32_t blacklistGpuCount);

    bool mapGpuIndexForOneNode(uint32_t nodeId,
                               DcgmFMGpuLWLinkConnMatrixList &connMatrix);

    bool updateGpuConnectionMatrixInfo(DcgmFMGpuLWLinkConnMatrixList &connMatrixList,
                                       DcgmDetailedEndPointInfo gpuEndPointInfo,
                                       TopologyLWLinkConnEndPoint topoGpuEndPoint,
                                       uint64 gpuEnabledLinkMask);

    bool getTopologyLWLinkConnGpuEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                          TopologyLWLinkConnEndPoint &gpuEndPoint);

    bool getTopologyLWLinkConnBySwitchEndPoint(TopologyLWLinkConnEndPoint &switchEndPoint,
                                               TopologyLWLinkConn &topoLWLinkConn);

    bool isLWLinkTrunkConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn);

    bool getAccessConnDetailedEndPointInfo(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                           DcgmDetailedEndPointInfo &switchEndPointInfo,
                                           DcgmDetailedEndPointInfo &gpuEndPointInfo);

    bool verifyAccessConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn);

    bool getLWSwitchInfoByLWLinkDevInfo(uint32 nodeId,
                                        DcgmFMLWLinkDevInfo &lwLinkDevInfo,
                                        DcgmFMLWSwitchInfo &lwSwitchInfo);

    bool getGpuInfoByLWLinkDevInfo(uint32 nodeId,
                                   DcgmFMLWLinkDevInfo &lwLinkDevInfo,
                                   DcgmFMGpuInfo &gpuInfo);

    bool getAccessConnGpuLinkMaskInfo(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                      uint64 &gpuEnabledLinkMask);

    int getGpuConnNum(uint32 nodeId, uint32_t physicalIdx);
    int getMaxGpuConnNum(void);

    bool isInterNodeTrunkConnsActive(void);

    void checkLWSwitchLinkInitStatus(void);
    void checkLWSwitchDeviceLinkInitStatus(uint32_t nodeId,
                                           DcgmFMLWLinkDevInfo &devInfo);
    void checkGpuLinkInitStatus(void);
    void checkGpuDeviceLinkInitStatus(uint32_t nodeId,
                                      DcgmFMLWLinkDevInfo &devInfo);

    uint32_t getInterNodeTrunkConnCount(uint32_t nodeId);

    DcgmGlobalFabricManager *mGfm;
};
