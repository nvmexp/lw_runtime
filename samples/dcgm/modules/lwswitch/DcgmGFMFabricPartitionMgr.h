#pragma once

/*****************************************************************************/
/*  Implement all the globalFM side LWSwitch Shared VM fabric partition      */
/*  related interfaces/methods.                                              */
/*****************************************************************************/

#include "DcgmFMCommon.h"
#include "dcgm_module_fm_structs_internal.h"
#include "lwos.h"
#include "fabricmanagerHA.pb.h"

class DcgmFMLWLinkConnRepo;
class DcgmGFMLWLinkDevRepo;
class DcgmFMLWLinkDetailedConnInfo;

//
// Place holder to cache dynamic information about GPUs like enumeration index,
// PCI BDF etc. This information can change every time the GPU are 
// attached/detached from the Driver/LWML or ServiceVM.
//
typedef struct {
    DcgmFMPciInfo pciInfo;
    uint32_t gpuIndex;
} PartitionGpuDynamicInfo;

typedef struct {
    uint32 physicalId;
    char uuid[DCGM_DEVICE_UUID_BUFFER_SIZE];
    uint32 numEnabledLinks;
    uint64 enabledLinkMask;
    PartitionGpuDynamicInfo dynamicInfo;
} PartitionGpuInfo;

typedef std::list<PartitionGpuInfo> PartitionGpuInfoList;

typedef struct {
    uint32 physicalId;
    uint32 numEnabledLinks;
    uint64 enabledLinkMask;
} PartitionSwitchInfo;

typedef std::list<PartitionSwitchInfo> PartitionSwitchInfoList;

typedef struct {
    uint32 partitionId;
    uint32 isActive;
    uint32 trunkConnCount;
    bool   errorHandled;
    PartitionGpuInfoList gpuInfo;
    PartitionSwitchInfoList switchInfo;
} PartitionInfo;

typedef std::list<PartitionInfo> PartitionInfoList;

#define PARTITION_CFG_TIMEOUT_MS 40000 // 40000ms, 40s
                                       // accommodate all GPU attach and detach
                                       // at partition configuration time

#define SHARED_FABRIC_PARTITION_HA_STATE_VER_1    1
#define SHARED_FABRIC_PARTITION_HA_STATE_VER      SHARED_FABRIC_PARTITION_HA_STATE_VER_1

class DcgmGFMFabricPartitionMgr
{
    friend class DcgmGlobalFMErrorHndlr;

public:
    DcgmGFMFabricPartitionMgr(DcgmGlobalFabricManager *pGfm);

    ~DcgmGFMFabricPartitionMgr();

    bool buildPartitionMappings();
    dcgmReturn_t getPartitions(dcgmFabricPartitionList_t &dcgmFabricPartitions);
    dcgmReturn_t activatePartition(uint32 nodeId, unsigned int partitionId);
    dcgmReturn_t deactivatePartition(uint32 nodeId, unsigned int partitionId);
    dcgmReturn_t setActivatedPartitions(dcgmActivatedFabricPartitionList_t &dcgmFabricPartitions);

    bool isPartitionConfigFailed(uint32 nodeId, uint32_t partitionId);
    void setPartitionConfigFailure(uint32 nodeId, unsigned int partitionId);
    void clearPartitionConfigFailure(uint32 nodeId, unsigned int partitionId);
    uint32_t getActivePartitionIdForLWSwitchPort(uint32_t nodeId, uint32_t physicalId, uint32_t portNum);

    bool getSharedFabricHaState(fabricmanagerHA::sharedFabricPartiontionInfo &haState);
    bool validateSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState);
    bool loadSharedFabricHaState(const fabricmanagerHA::sharedFabricPartiontionInfo &savedHaState);

    bool isInitDone(void) { return mInitDone; };

private:
    bool getSharedLWSwitchPartitionInfo(uint32 nodeId, unsigned int partitionId, PartitionInfo &partInfo);
    bool mapTopologyPartition(uint32_t nodeId, const sharedLWSwitchPartitionInfo &topoPartInfo);
    bool isPartitionExists(uint32 nodeId, unsigned int partitionId);
    void setPartitionActiveState(uint32 nodeId, unsigned int partitionId, bool active);
    bool isPartitionActive(uint32 nodeId, unsigned int partitionId);
    bool validatePartitionGpus(uint32 nodeId, unsigned int partitionId,
                               DcgmFMGpuInfoMap &gpuInfoMap);
    bool validatePartitionLWLinkConns(uint32 nodeId, unsigned int partitionId,
                                      DcgmFMLWLinkConnRepo &lwLinkConnRepo);
    void updatePartitionGpuDynamicInfo(uint32 nodeId, unsigned int partitionId,
                                       DcgmFMGpuInfoMap &gpuInfoMap);
    bool resetPartitionLWSwitchLinks(uint32 nodeId, unsigned int partitionId, bool inErrHdlr);
    void filterPartitionLWLinkTrunkConns(uint32 nodeId, unsigned int partitionId,
                                         DcgmFMLWLinkConnRepo &linkConnRepo,
                                         DcgmGFMLWLinkDevRepo &linkDevRepo);
    bool partitionIsLWLinkTrunkConnection(DcgmFMLWLinkDetailedConnInfo *lwLinkConn,
                                         DcgmGFMLWLinkDevRepo &linkDevRepo);
    void partitionResetFilteredTrunkLWLinks(uint32 nodeId, unsigned int partitionId, bool inErrHdlr);
    void partitionPowerOffTrunkLWLinkConns(uint32 nodeId, unsigned int partitionId, bool inErrHdlr);
    void handlePartitionConfigFailure(uint32 nodeId, unsigned int partitionId);
    void getGpusUsedInActivatePartitions(uint32_t nodeId, std::set<uint32_t> &usedGpus);
    bool isGpuUsedInActivePartitions(uint32_t nodeId, uint32_t partitionId);

    // list of all the supported partitions and its comprehensive information
    PartitionInfoList mSupportedPartitions;
    DcgmGlobalFabricManager *mGfm;
    LWOSCriticalSection mLock;
    bool mInitDone;

    fabricmanagerHA::sharedFabricPartiontionInfo *mpHaState;
};
