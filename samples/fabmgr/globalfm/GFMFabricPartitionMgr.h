/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
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
/*  Implement all the globalFM side LWSwitch Shared VM fabric partition      */
/*  related interfaces/methods.                                              */
/*****************************************************************************/

#include "FMCommonTypes.h"
#include "lwos.h"
#include "lw_fm_types.h"
#include "fabricmanagerHA.pb.h"
#include "GlobalFmFabricParser.h"

class GlobalFMLWLinkConnRepo;
class GlobalFMLWLinkDevRepo;
class FMLWLinkDetailedConnInfo;
class FMLWLinkDevInfo;
//
// Place holder to cache dynamic information about GPUs like enumeration index,
// PCI BDF, GFID, GFID Mask etc. This information can change every time the GPU
// are attached/detached from the Driver/LWML or ServiceVM.
//
typedef struct {
    uint32_t gfid;
    uint32_t gfidMask;
} GpuGfidInfo;

typedef struct {
    FMPciInfo_t pciInfo;
    uint32_t gpuIndex;
    GpuGfidInfo gfidInfo;
} PartitionGpuDynamicInfo;

typedef struct {
    uint32 physicalId;
    char uuid[FM_UUID_BUFFER_SIZE];
    uint32 numEnabledLinks;     // number of Links to be enabled for the partition
    uint64 enabledLinkMask;     // mask of links to be enabled
    uint32 numLinksAvailable;   // number of maximum active/working links after FM initialization
    uint32 linkLineRateMBps;    // per link speed
    uint32 discoveredLinkMask;  // discovered (aka supported) links mask on this GPU
    PartitionGpuDynamicInfo dynamicInfo;
} PartitionGpuInfo;

typedef std::list<PartitionGpuInfo> PartitionGpuInfoList;

typedef struct {
    uint32 physicalId;
    uint32 archType;
    uint32 numEnabledLinks;
    uint64 enabledLinkMask;
} PartitionSwitchInfo;

typedef std::list<PartitionSwitchInfo> PartitionSwitchInfoList;

typedef struct {
    uint32 partitionId;
    uint32_t partitionState;
    uint32 trunkConnCount;
    bool   errorOclwrred;
    PartitionGpuInfoList gpuInfo;
    PartitionSwitchInfoList switchInfo;
} PartitionInfo;

//
// Partition States
//
// Transition: DEACTIVE <-> ACTIVE <-> SYNC_PENDING ->DEACTIVE
//
// PARTITION_IN_SYNC_PENDING_STATE:
//     During FM restart, all the previously activated partitions will be
//     set to SYNC_PENDING state based on the FM HA state file. During
//     setActivatedFabricPartitions() API, these partitions will be changed
//     either to ACTIVE or DEACTIVE state based on the parition list that
//     will be provided by caller.
//
typedef enum {
    PARTITION_IN_DEACTIVE_STATE = 0x0,      // Partition is in deactive state
    PARTITION_IN_ACTIVE_STATE = 0x1,        // Partition is in active state
    PARTITION_IN_SYNC_PENDING_STATE         // Partition is in sync pending state
} FABRIC_PARTITION_STATE_TYPE;

typedef std::list<PartitionInfo> PartitionInfoList;

#define FABRIC_PARTITION_HA_STATE_VER_1    1
#define FABRIC_PARTITION_HA_STATE_VER      FABRIC_PARTITION_HA_STATE_VER_1

class GlobalFMFabricPartitionMgr
{
    friend class GlobalFMErrorHndlr;

public:
    GlobalFMFabricPartitionMgr(GlobalFabricManager *pGfm);

    ~GlobalFMFabricPartitionMgr();

    bool buildPartitionMappings();
    FMIntReturn_t getPartitions(fmFabricPartitionList_t &fmFabricPartitionList);
    FMIntReturn_t activatePartition(uint32 nodeId, fmFabricPartitionId_t partitionId);
    FMIntReturn_t activatePartitionWithVfs(uint32 nodeId, fmFabricPartitionId_t partitionId, fmPciDevice_t *vfList, unsigned int numVfs);
    FMIntReturn_t deactivatePartition(uint32 nodeId, fmFabricPartitionId_t partitionId);
    FMIntReturn_t setActivatedPartitions(fmActivatedFabricPartitionList_t &fmFabricPartitions);
    FMIntReturn_t getUnsupportedPartitions(fmUnsupportedFabricPartitionList_t &fmFabricPartitionList);

    bool isPartitionConfigFailed(uint32 nodeId, uint32_t partitionId);
    bool isPartitionActive(uint32 nodeId, uint32_t partitionId);
    void setPartitionConfigFailure(uint32 nodeId, unsigned int partitionId);
    void clearPartitionConfigFailure(uint32 nodeId, unsigned int partitionId);
    uint32_t getActivePartitionIdForLWSwitchPort(uint32_t nodeId, uint32_t physicalId, uint32_t portNum);

    bool getSharedFabricHaState(fabricmanagerHA::sharedFabricPartiontionInfo &haState);
    bool validateSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState);
    bool loadSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState);

    bool isGpuInActivePartition(uint32 nodeId, char *gpuUuid);
    bool isInitDone(void) { return mInitDone; };
    void addGpuToResetPendingList(char *uuid);
    void remGpuFromResetPendingList(char *uuid);

    bool getSwitchEnabledLinkMaskForPartition(uint32_t partitionId, SwitchKeyType switchKey, uint64_t &enabledLinkMask);
    bool isGpuUsedInPartition(uint32_t partitionId, GpuKeyType switchKey);

private:
    bool getSharedLWSwitchPartitionInfo(uint32 nodeId, unsigned int partitionId, PartitionInfo &partInfo);
    bool mapTopologyPartition(uint32_t nodeId, const sharedLWSwitchPartitionInfo &topoPartInfo);
    bool isPartitionExists(uint32 nodeId, unsigned int partitionId);
    uint32_t getPartitionActiveState(uint32 nodeId, unsigned int partitionId);
    void setPartitionActiveState(uint32 nodeId, unsigned int partitionId, bool active);
    bool validatePartitionGpus(uint32 nodeId, unsigned int partitionId,
                               FMGpuInfoMap &gpuInfoMap);
    bool validatePartitionLWLinkConns(uint32 nodeId, unsigned int partitionId,
                                      GlobalFMLWLinkConnRepo &lwLinkConnRepo,
                                      FMGpuInfoMap &gpuInfoMap,
                                      GlobalFMLWLinkDevRepo &lwLinkDevRepo);
    void updatePartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId,
                                       FMGpuInfoMap &gpuInfoMap);
    void setGfidInPartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId, std::list<GpuGfidInfo> &gfidList);
    void clearGfidInPartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId);
    bool resetPartitionLWSwitchLinks(uint32 nodeId, unsigned int partitionId);
    void filterPartitionLWLinkTrunkConns(uint32 nodeId, unsigned int partitionId,
                                         GlobalFMLWLinkConnRepo &linkConnRepo,
                                         GlobalFMLWLinkDevRepo &linkDevRepo);
    bool partitionIsLWLinkTrunkConnection(FMLWLinkDetailedConnInfo *lwLinkConn,
                                         GlobalFMLWLinkDevRepo &linkDevRepo);
    void partitionResetFilteredTrunkLWLinks(uint32 nodeId, unsigned int partitionId);
    void partitionPowerOffTrunkLWLinkConns(uint32 nodeId, unsigned int partitionId);
    void handlePartitionActivationError(uint32_t nodeId, PartitionInfo &partInfo);
    void getGpusUsedInActivatePartitions(uint32_t nodeId, std::set<uint32_t> &usedGpus);
    bool isGpuUsedInActivePartitions(uint32_t nodeId, uint32_t partitionId);
    bool isAnotherPartitionWithTrunkLinksActive(uint32_t nodeId, uint32_t partitionId);
    uint32_t getGpuNumEnabledLinks(uint32_t nodeId, char *gpuUuid, FMGpuInfoMap &gpuInfoMap);
    bool isGpuOnNotDetectedBasebard(uint32_t nodeId, uint32_t physicalId);
    void populateUnsupportedPartitions(void);

    FMIntReturn_t sendActivationTrainingFailedInfoForSwitches(uint32 nodeId,
                                                              unsigned int partitionId,
                                                              GlobalFMLWLinkDevRepo &lwLinkDevRepo);
    FMIntReturn_t sendActivationTrainingFailedInfoForSwitch(uint32_t nodeId,
                                                            PartitionSwitchInfo &partSwitchInfo,
                                                            FMLWLinkDevInfo &switchLwLinkDevInfo);
    bool isGpuResetInProgress(char *uuid);

    // list of all the supported partitions and its comprehensive information
    PartitionInfoList mSupportedPartitions;
    fmUnsupportedFabricPartitionList_t mUnsupportedPartitionList;
    GlobalFabricManager *mGfm;
    LWOSCriticalSection mLock;
    bool mInitDone;

    // list of gpus that are in reset
    typedef std::set <FMUuid_t> ResetGpusByUuid;
    ResetGpusByUuid mResetPendingGpus;
};

