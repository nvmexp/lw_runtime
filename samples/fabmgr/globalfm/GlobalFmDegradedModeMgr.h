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
 * determine the actions when LWSwitch, LWSwitch access link, trunk link, and
 * GPU fails.
 */

#include "FMCommonTypes.h"
#include "GlobalFabricManager.h"
#include "GlobalFmFabricParser.h"
#include "FMLWLinkDeviceRepo.h"
#include "fabricmanager.pb.h"
#include "FMTopologyValidator.h"

class FMFabricParser;
class GlobalFabricManager;
class FMTopologyValidator;


typedef std::map <SwitchKeyType, lwswitch::SwitchDegradedReason> DegradedLWSwitchMap;
typedef std::map <GpuKeyType, lwswitch::GpuDegradedReason> DegradedGpuMap;
typedef std::set <PartitionKeyType> PartitionSet;
typedef std::set <FMUuid_t> DegradedGpusByUuid;

class GlobalFmDegradedModeMgr
{
public:
    typedef enum {
        GPU_LINK_FAILURE_DISABLE_GPU        = 0,
        GPU_LINK_FAILURE_DISABLE_LWSWITCH
    } GpuLinkFailureMode;

    typedef enum {
        LWSWITCH_TRUNK_LINK_FAILURE_ABORT_FM = 0,
        LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_LWSWITCH,
        LWSWITCH_TRUNK_LINK_FAILURE_DISABLE_PARTITION
    } LwswitchTrunkLinkFailureMode;

    typedef enum {
        LWSWITCH_FAILURE_ABORT_FM           = 0,
        LWSWITCH_FAILURE_DISABLE_LWSWITCH,
        LWSWITCH_FAILURE_DISABLE_PARTITION
    } LwswitchFailureMode;

    GlobalFmDegradedModeMgr(GlobalFabricManager *pGfm,
                            uint32_t accessLinkFailureMode,
                            uint32_t trunkLinkFailureMode,
                            uint32_t lwswitchFailureMode);

    ~GlobalFmDegradedModeMgr();

    void addFailedSwitch(uint32_t nodeId, uint32_t physicalId);

    DegradedLWSwitchMap mDegradedSwitches;
    DegradedGpuMap mDegradedGpus;
    PartitionSet mDisabledPartitions;
    // this lookup is needed for gpu reset as it is not safe to do lookup based on physical id
    DegradedGpusByUuid mDegradedGpusByUuid;

    bool isSwitchDegraded(uint32_t nodeId, uint32_t physicalId,
                          lwswitch::SwitchDegradedReason &reason);

    bool isGpuDegraded(uint32_t nodeId, uint32_t physicalId,
                       lwswitch::GpuDegradedReason &reason);

    bool isPartitionDisabled(uint32_t nodeId, uint32_t partitionId);

    void handleMessage( lwswitch::fmMessage *pFmMessage);
    void sendAllDegradedGpuInfo(void);
    void sendAllDegradedLwswitchInfo(void);

    void processFailures();

    bool isAnyDeviceDegraded();
    bool isGpuDegraded(char *uuid);

    void getLwlinkFailedDevices(uint32_t nodeId, fmLwlinkFailedDevices_t *devList);
    int getNumDegradedLinksForGpu();

private:

    void getAllFailedLwlinks(void);
    void getFailedLwlinksFromSwitch(uint32_t nodeId, FMLWLinkDevInfo &devInfo);
    void getFailedLwlinksFromGpu(uint32_t nodeId, FMLWLinkDevInfo &devInfo);

    void degradeOneSwitch(SwitchKeyType &key, lwswitch::SwitchDegradedReason reason);
    void degradeOneGpu(GpuKeyType &key, lwswitch::GpuDegradedReason reason);

    void processSwitchFailures();
    void processTrunkLinkFailures();
    void processAccessLinkFailures();
    void processExcludedSwitches();

    void sendDegradedInfoToAllNodes();

    int turnOffLwlinksOnDegradedDev(GlobalFMLWLinkIntf *linkTrainIntf,
                                    GlobalFMLWLinkConnRepo &linkConnRepo,
                                    GlobalFMLWLinkDevRepo &linkDevRepo);

    void sendDegradedGpuInfo(uint32_t nodeId);
    void sendDegradedLwswitchInfo(uint32_t nodeId);

    void handleDegradeGpuInfoAckMsg(lwswitch::fmMessage *pFmMessage);
    void handleDegradeLwswitchInfoAckMsg(lwswitch::fmMessage *pFmMessage);
    void dumpMessage(lwswitch::fmMessage *pFmMessage);
    FMIntReturn_t SendMessageToLfm(uint32_t nodeId, lwswitch::fmMessage *pFmMessage);
    void getPeerReason(lwswitch::SwitchDegradedReason reason, lwswitch::SwitchDegradedReason &peerSwitchReason);
    const char* getReasonAsString(lwswitch::SwitchDegradedReason reason);
    bool isMultipleLWSwitchesExcluded(uint32_t nodeId, FMExcludedLWSwitchInfoList &excludedInfoList);

    void getLwlinkFailedGpus(uint32_t nodeId, uint32_t &numGpus, fmLwlinkFailedDeviceInfo_t gpuInfo[]);
    void getLwlinkFailedSwitches(uint32_t nodeId, uint32_t &numSwitches, fmLwlinkFailedDeviceInfo_t switchInfo[]);
    void removeLinkFromMissingConnLinkList(uint32 linkIndex, std::list<uint32> &missingConnLinkIndex);
    void updateSwitchFailedTrunkLinkInfoByPeerSwitch(SwitchKeyType switchKey, SwitchKeyType peerSwitchKey,
                                                     std::list<uint32_t> &peerFailedTrunkPortList);

    void populateLwlinkFailedDeviceMap(uint32_t nodeId);
    void buildDegradedGpuUuidLookup();
    int getNumPairsSwitchDegraded();

    GlobalFabricManager *mpGfm;

    typedef std::list<uint32_t> FailedPortList;
    std::set <SwitchKeyType> mFailedSwitch;
    std::map <SwitchKeyType, FailedPortList> mSwitchWithFailedTrunkPorts;
    std::map <SwitchKeyType, FailedPortList> mSwitchWithFailedAccessPorts;
    std::map <SwitchKeyType, FailedPortList> mSwitchWithFailedPorts;
    std::map <GpuKeyType, FailedPortList> mGpuWithFailedAccessPorts;
    std::map <SwitchKeyType, vector<SwitchKeyType>> mSwitchPairs;
    std::set <GpuKeyType> mGpuWithAllLinksContain;

    GpuLinkFailureMode mGpuLinkFailureMode;
    LwswitchTrunkLinkFailureMode mLwswitchTrunkLinkFailureMode;
    LwswitchFailureMode mLwswitchFailureMode;

    std::map <uint32_t, fmLwlinkFailedDevices_t> mLwlinkFailedDeviceMap;


    bool mAbortFm;
};
