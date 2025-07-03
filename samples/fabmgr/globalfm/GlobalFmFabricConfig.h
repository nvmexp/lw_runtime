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

#include <iostream>
#include <fstream>
#include <string>

#include "FMCommCtrl.h"
#include "GlobalFmErrorHndlr.h"
#include "GFMFabricPartitionMgr.h"
#include "FMErrorCodesInternal.h"
#include "GlobalFmFabricParser.h"

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#include "GlobalFmMulticastMgr.h"
#endif

class GlobalFabricManager;
class FMFabricParser;
struct NodeConfig;

class FMFabricConfig : public FMMessageHandler
{
public:
    
    FMFabricConfig(GlobalFabricManager *pGfm);
    virtual ~FMFabricConfig();

    virtual void handleEvent( FabricManagerCommEventType eventType, uint32_t nodeId );

    FMIntReturn_t sendNodeGlobalConfig( uint32_t nodeId );
    FMIntReturn_t sendPeerLFMInfo( uint32_t nodeId );
    FMIntReturn_t configOneNode( uint32_t nodeId, std::set<enum PortType> &portTypes );
    FMIntReturn_t configOneLwswitch( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   std::set<enum PortType> &portTypes );
    FMIntReturn_t configSwitchPorts( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                     std::set<enum PortType> &portTypes );
    FMIntReturn_t configSwitchPortList( uint32_t nodeId, uint32_t partitionId,
                                        std::list<PortKeyType> &portList, bool sync );
    FMIntReturn_t configSwitchPortsWithTypes( std::set<enum PortType> &portTypes );
    FMIntReturn_t configLwswitches( uint32_t nodeId, std::set<enum PortType> &portTypes );
    FMIntReturn_t configGpus( uint32_t nodeId );

    FMIntReturn_t configIngressReqEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FMIntReturn_t configIngressRespEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FMIntReturn_t configGangedLinkEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );

    FMIntReturn_t configIngressReqTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FMIntReturn_t configIngressRespTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FMIntReturn_t configGangedLinkTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );

    FMIntReturn_t configGetGfidForPartition(uint32_t nodeId, PartitionInfo &partInfo, fmPciDevice_t *vfList,
                                            std::list<GpuGfidInfo> &gfidList);
    FMIntReturn_t configCfgGfidForPartition(uint32_t nodeId, PartitionInfo &partInfo, bool activate);
    FMIntReturn_t configActivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo );
    FMIntReturn_t configDeactivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo );

    FMIntReturn_t configSetGpuDisabledLinkMaskForPartition( uint32_t nodeId, PartitionInfo &partInfo );
    FMIntReturn_t configSharedLWSwitchPartitionAttachGPUs( uint32_t nodeId, PartitionInfo &partInfo );
    FMIntReturn_t configSharedLWSwitchPartitionDetachGPUs( uint32_t nodeId, PartitionInfo &partInfo );
    FMIntReturn_t configDetachAllGPUs( uint32_t nodeId );

    FMIntReturn_t configAttachGpu( uint32_t nodeId, char *gpuUuid );
    FMIntReturn_t configDetachGpu( uint32_t nodeId, char *gpuUuid );
    FMIntReturn_t configGpu( uint32_t nodeId, char *gpuUuid );

    void removeDisabledPortsFromRidEntry( uint32_t nodeId, uint32_t physicalId, ridRouteEntry *entry );
    void removeDisabledPortsFromRlanEntry( uint32_t nodeId, uint32_t physicalId, rlanRouteEntry *entry );

    FMIntReturn_t configRmapEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex, bool sync );
    FMIntReturn_t configRidEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex, bool sync );
    FMIntReturn_t configRlanEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex, bool sync );

    FMIntReturn_t configRmapTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FMIntReturn_t configRidTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FMIntReturn_t configRlanTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    FMIntReturn_t configMulticastRoutes( uint32_t nodeId, uint32_t switchPhysicalId,
                                         MulticastGroupInfo *groupInfo, PortMulticastTable &portMcastTable,
                                         bool freeGroup, bool sync );

    FMIntReturn_t configMulticastRemapTable( uint32_t nodeId, uint32_t switchPhysicalId,
                                             MulticastGroupInfo *groupInfo, PortMulticastTable &portMcastTable,
                                             uint64_t mappedAddr, bool freeGroup, bool sync );
#endif

    FMIntReturn_t sendConfigInitDoneReqMsg( uint32_t nodeId );
    FMIntReturn_t sendConfigDeInitReqMsg( uint32_t nodeId );

    FMIntReturn_t configDisableSwitchTrunkLinks( uint32_t nodeId );

    void handleMessage( lwswitch::fmMessage *pFmMessage);
    void handleConfigError( uint32_t nodeId, GlobalFMErrorSource errSource,
                            GlobalFMErrorTypes errType, lwswitch::fmMessage &errMsg );
    void handleSharedLWSwitchPartitionConfigError( uint32_t nodeId, uint32_t partitionId,
                                                   GlobalFMErrorSource errSource,
                                                   GlobalFMErrorTypes errType, lwswitch::fmMessage &errMsg );

    bool isPendingConfigReqEmpty( uint32_t nodeId, uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );
    void dumpPendingConfigReq( uint32_t nodeId, uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );
    void clearPartitionPendingConfigRequest( uint32_t nodeId, uint32_t partitionId );

private:

    typedef struct ConfigRequestKeyType
    {
        uint32_t nodeId;
        uint32_t partitionId;
        uint32_t requestId;

        bool operator<(const ConfigRequestKeyType &k) const {
            return ( (nodeId < k.nodeId) ||
                     ( (nodeId ==  k.nodeId) && (partitionId < k.partitionId) ) ||
                     ( (nodeId ==  k.nodeId) && (partitionId == k.partitionId) && (requestId <  k.requestId) ) );
        }
    } ConfigRequestKeyType;

    typedef std::pair<uint32_t, uint32_t> GpuPhyIdToIndexPair;
    typedef std::list<GpuPhyIdToIndexPair> GpuPhyIdToIndexList;

    GlobalFabricManager            *mpGfm;                // global fabric manager
    uint32_t                        mCfgMsgTimeoutSec;

    FMIntReturn_t sendSwitchDisableLinkReq( uint32_t nodeId, uint32_t physicalId, uint64 disableMask);

    FMIntReturn_t SendMessageToLfm( uint32_t nodeId, uint32_t partitionId,
                                    lwswitch::fmMessage *pFmMessage, bool trackReq );

    FMIntReturn_t SendMessageToLfmSync( uint32_t nodeId, uint32_t partitionId,
                                        lwswitch::fmMessage *pFmMessage, lwswitch::fmMessage **pResponse,
                                        uint32_t timeoutSec );

    void dumpMessage( lwswitch::fmMessage *pFmMessage );

    void addPendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                  uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );

    void removePendingConfigRequest ( uint32_t nodeId, uint32_t requestId,
                                      uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );

    bool handleNodeGlobalConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleSWPortConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

    bool handleIngReqTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleIngRespTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleGangedLinkTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

    bool handleGpuAttachRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

    bool handleGpuDetachRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

    bool handleConfigInitDoneRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleConfigNodeInfoAckMsg( lwswitch::fmMessage *pFmMessage );

    bool handleConfigDeinitRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleConfigSwitchDisableLinkRespMsg( lwswitch::fmMessage *pFmMessage );

    bool handleConfigGpuSetDisabledLinkMaskRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

    bool handleCommonGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage,
                                       const lwswitch::gpuConfigResponse *respMsg, bool handleErr );
    bool handleGetGfidRespMsg(lwswitch::fmMessage *pFmMessage, std::list<GpuGfidInfo> &gfidList);
    bool handleCfgGfidRespMsg(lwswitch::fmMessage *pFmMessage);

    bool handleRmapTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );
    bool handleRidTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );
    bool handleRlanTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool handleMulticastConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr );
#endif

    lwswitch::fmMessage *generateConfigGpusMsg( uint32_t nodeId, uint32_t partitionId,
                                                PartitionInfo &partInfo,
                                                std::list<uint32_t> &gpuPhysicalIdList,
                                                lwswitch::FabricManagerMessageType configGPUMsgType,
                                                bool activate );

    FMIntReturn_t configGpus( uint32_t nodeId, uint32_t patitionId, PartitionInfo &partInfo,
                              std::list<uint32_t> gpuPhysicalIdList, bool activate );

    FMIntReturn_t attachGpus( uint32_t nodeId, PartitionInfo &partInfo );

    /* 
        partInfo is defaulted to NULL as there are cases where are we need to detach All gpus
        as opposed to per GPU detach. There are calls from the GlobalFabricManager.cpp and 
        GlobalFMErrorHndl.cpp where we have cases to detach all GPUs.
    */
    FMIntReturn_t detachGpus( uint32_t nodeId, PartitionInfo *partInfo = NULL);

    FMIntReturn_t configIngressReqTable( uint32_t nodeId, uint32_t partitionId,
                                         uint32_t switchPhysicalId,  uint32_t portNum,
                                         std::list<uint32_t> &gpuPhysicalIds, bool activate);

    FMIntReturn_t configIngressRespTable( uint32_t nodeId, uint32_t partitionId,
                                          uint32_t switchPhysicalId,  uint32_t portNum,
                                          std::list<uint32_t> &gpuPhysicalIds, bool activate);

    FMIntReturn_t configRmapTable( uint32_t nodeId, uint32_t partitionId,
                                   uint32_t switchPhysicalId,  uint32_t portNum,
                                   std::list<uint32_t> &gpuPhysicalIds, bool activate );
    FMIntReturn_t configRmapTableWithVfs(uint32_t nodeId, uint32_t partitionId,
                                         uint32_t switchPhysicalId,  uint32_t portNum,
                                         PartitionInfo &partInfo, bool activate );
    FMIntReturn_t configRidTable( uint32_t nodeId, uint32_t partitionId,
                                  uint32_t switchPhysicalId,  uint32_t portNum,
                                  std::list<uint32_t> &gpuPhysicalIds, bool activate );
    FMIntReturn_t configRlanTable( uint32_t nodeId, uint32_t partitionId,
                                   uint32_t switchPhysicalId,  uint32_t portNum,
                                   std::list<uint32_t> &gpuPhysicalIds, bool activate );

    FMIntReturn_t configRmapTableEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                          int portIndex, bool sync );

    rmapPolicyEntry *getRmapEntryByTable( RmapTableKeyType key, RemapTable rmapTable );

    FMIntReturn_t configRmapTableByAddrType( uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                             uint32_t portNum, std::list<uint32_t> &gpuPhysicalIds, bool activate,
                                             FabricAddrType addrType);

    FMIntReturn_t configRmapTableEntriesByTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                                 int portIndex, bool sync, RemapTable rmapTable );

    FMIntReturn_t configSharedLWSwitchPartitionRoutingTable( uint32_t nodeId, PartitionInfo &partInfo, bool activate );
    
    FMIntReturn_t configSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                                 bool activate );

    FMIntReturn_t configSharedLWSwitchPartitionGPUs( uint32_t nodeId, PartitionInfo &partInfo, bool activate );

    void removeDisabledPortsFromRoutingEntry( uint32_t nodeId, uint32_t physicalId,
                                              int32_t *vcmodevalid7_0, int32_t *vcmodevalid15_8,
                                              int32_t *vcmodevalid17_16, int32_t *entryValid );

    void removeDisabledPortsFromIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                                 ingressRequestTable *entry );

    void removeDisabledPortsFromIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                                  ingressResponseTable *entry );

    bool waitForPartitionConfigCompletion(uint32_t nodeId, uint32_t partitionId,
                                          uint32_t timeoutMs);

    std::map <ConfigRequestKeyType, unsigned int> mPendingConfigReqMap;
    LWOSCriticalSection mLock;
};


