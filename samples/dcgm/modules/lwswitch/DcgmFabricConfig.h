#ifndef DCGM_FABRIC_CONFIG_H
#define DCGM_FABRIC_CONFIG_H

#include <iostream>
#include <fstream>
#include <string>

#include "DcgmFMCommCtrl.h"
#include "DcgmGlobalFMErrorHndlr.h"
#include "DcgmGFMFabricPartitionMgr.h"
#include "DcgmFMError.h"

class DcgmGlobalFabricManager;
class DcgmFabricParser;
struct NodeConfig;

class DcgmFabricConfig : public FMMessageHandler
{
public:
    
    DcgmFabricConfig(DcgmGlobalFabricManager *pGfm);
    virtual ~DcgmFabricConfig();

    virtual void handleEvent( FabricManagerCommEventType eventType, uint32_t nodeId );

    FM_ERROR_CODE sendNodeGlobalConfig( uint32_t nodeId );
    FM_ERROR_CODE sendPeerLFMInfo( uint32_t nodeId );
    FM_ERROR_CODE configOneNode( uint32_t nodeId, std::set<enum PortType> &portTypes );
    FM_ERROR_CODE configOneLwswitch( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   std::set<enum PortType> &portTypes );
    FM_ERROR_CODE configSwitchPorts( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                     std::set<enum PortType> &portTypes );
    FM_ERROR_CODE configSwitchPortsWithTypes( std::set<enum PortType> &portTypes );
    FM_ERROR_CODE configLwswitches( uint32_t nodeId, std::set<enum PortType> &portTypes );
    FM_ERROR_CODE configGpus( uint32_t nodeId );

    FM_ERROR_CODE configIngressReqEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FM_ERROR_CODE configIngressRespEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FM_ERROR_CODE configGangedLinkEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );

    FM_ERROR_CODE configIngressReqTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FM_ERROR_CODE configIngressRespTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FM_ERROR_CODE configGangedLinkTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );

    FM_ERROR_CODE configActivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo );
    FM_ERROR_CODE configDeactivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                                           bool inErrHdlr = false );
    FM_ERROR_CODE configSetGpuDisabledLinkMaskForPartition( uint32_t nodeId, PartitionInfo &partInfo );
    FM_ERROR_CODE configSharedLWSwitchPartitionAttachGPUs( uint32_t nodeId, uint32_t partitionId );
    FM_ERROR_CODE configSharedLWSwitchPartitionDetachGPUs( uint32_t nodeId, uint32_t partitionId,
                                                           bool inErrHdlr = false );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void removeDisabledPortsFromRidEntry( uint32_t nodeId, uint32_t physicalId, ridRouteEntry *entry );
    void removeDisabledPortsFromRlanEntry( uint32_t nodeId, uint32_t physicalId, rlanRouteEntry *entry );

    FM_ERROR_CODE configRmapEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FM_ERROR_CODE configRidEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );
    FM_ERROR_CODE configRlanEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex );

    FM_ERROR_CODE configRmapTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FM_ERROR_CODE configRidTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
    FM_ERROR_CODE configRlanTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo );
#endif

    FM_ERROR_CODE sendConfigInitDoneReqMsg( uint32_t nodeId );
    FM_ERROR_CODE sendConfigDeInitReqMsg( uint32_t nodeId );

    FM_ERROR_CODE configDisableSwitchLinks( uint32_t nodeId, uint64 disableMask );

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

    DcgmGlobalFabricManager            *mpGfm;                // global fabric manager

    FM_ERROR_CODE sendSwitchDisableLinkReq( uint32_t nodeId, uint32_t physicalId, uint64 disableMask);

    FM_ERROR_CODE SendMessageToLfm( uint32_t nodeId, uint32_t partitionId,
                                    lwswitch::fmMessage *pFmMessage, bool trackReq );

    void dumpMessage( lwswitch::fmMessage *pFmMessage );

    void addPendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                  uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );

    void removePendingConfigRequest ( uint32_t nodeId, uint32_t requestId,
                                      uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID );

    void handleNodeGlobalConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleSWPortConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleIngReqTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleIngRespTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleGangedLinkTblConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuAttachRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuDetachRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigInitDoneRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigNodeInfoAckMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigDeinitRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigSwitchDisableLinkRespMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigGpuSetDisabledLinkMaskRespMsg( lwswitch::fmMessage *pFmMessage );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void handleRmapTableConfigRespMsg( lwswitch::fmMessage *pFmMessage );
    void handleRidTableConfigRespMsg( lwswitch::fmMessage *pFmMessage );
    void handleRlanTableConfigRespMsg( lwswitch::fmMessage *pFmMessage );
#endif

    lwswitch::fmMessage *generateConfigGpusMsg( uint32_t nodeId, uint32_t partitionId,
                                                GpuPhyIdToIndexList &gpuList,
                                                lwswitch::FabricManagerMessageType configGPUMsgType );

    FM_ERROR_CODE configGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList);

    FM_ERROR_CODE attachGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList);

    FM_ERROR_CODE detachGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList);

    FM_ERROR_CODE configIngressReqTable( uint32_t nodeId, uint32_t patitionId,
                                         uint32_t switchPhysicalId,  uint32_t portNum,
                                         std::list<uint32_t> &gpuPhysicalIds, bool activate);

    FM_ERROR_CODE configIngressRespTable( uint32_t nodeId, uint32_t patitionId,
                                          uint32_t switchPhysicalId,  uint32_t portNum,
                                          std::list<uint32_t> &gpuPhysicalIds, bool activate);

    FM_ERROR_CODE configSharedLWSwitchPartitionRoutingTable( uint32_t nodeId, PartitionInfo &partInfo, bool activate );
    
    FM_ERROR_CODE configSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                                 bool activate, bool inErrHdlr = false );

    FM_ERROR_CODE configSharedLWSwitchPartitionGPUs( uint32_t nodeId, PartitionInfo &partInfo );

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

#endif
