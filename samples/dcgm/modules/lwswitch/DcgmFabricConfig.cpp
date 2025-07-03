#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include <google/protobuf/text_format.h>
#include "logging.h"
#include "DcgmFabricConfig.h"
#include "DcgmLogging.h"
#include "DcgmFabricParser.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFMAutoLock.h"

DcgmFabricConfig::DcgmFabricConfig(DcgmGlobalFabricManager *pGfm)
{
    mpGfm = pGfm;

    // lock is required as the pending config request map will be accessed from
    // request message send, and response message handling
    lwosInitializeCriticalSection( &mLock );
};

DcgmFabricConfig::~DcgmFabricConfig()
{
    lwosDeleteCriticalSection( &mLock );
};

void
DcgmFabricConfig::handleEvent( FabricManagerCommEventType eventType, uint32_t nodeId )
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // There is no peer FM connection for config, not applicable.
            PRINT_DEBUG("%d", "nodeId %d FM_EVENT_PEER_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            // There is no peer FM connection for config, not applicable.
            PRINT_DEBUG("%d", "nodeId %d FM_EVENT_PEER_FM_DISCONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // configs are initiated from GFM. So not applicable for GFM itself
            PRINT_DEBUG("%d", "nodeId %d FM_EVENT_GLOBAL_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // configs are initiated from GFM. So not applicable for GFM itself
            PRINT_DEBUG("%d", "nodeId %d FM_EVENT_GLOBAL_FM_DISCONNECT", nodeId);
            break;
        }
    }
}

// TODO: need to check against restart,
//       and make sure the system is not re-initialized at restart.
FM_ERROR_CODE
DcgmFabricConfig::configOneNode( uint32_t nodeId, std::set<enum PortType> &portTypes )
{
    FM_ERROR_CODE rc;

    PRINT_DEBUG("%d", "nodeId %d", nodeId);

    rc = configLwswitches( nodeId, portTypes );
    if ( rc != FM_SUCCESS )
    {
        return rc;
    }

    rc = configGpus( nodeId );
    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::sendNodeGlobalConfig( uint32_t nodeId )
{
    FM_ERROR_CODE rc;
    lwswitch::nodeGlobalConfigRequest *pGlobalConfigReq = NULL;
    lwswitch::fmMessage *pMessage = NULL;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_NODE_GLOBAL_CONFIG_REQ );

    pGlobalConfigReq = new lwswitch::nodeGlobalConfigRequest;
    pGlobalConfigReq->set_localnodeid( nodeId );
    pMessage->set_allocated_globalconfigrequest( pGlobalConfigReq );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d",
                    "Failed to send Node Global Config to nodeId %d, return %d\n",
                    nodeId, rc);
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configOneLwswitch( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   std::set<enum PortType> &portTypes )
{
    FM_ERROR_CODE rc;

    rc = configSwitchPorts( nodeId, pSwitchInfo, portTypes );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed ports config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

    rc = configIngressReqTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed ingress request table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

    rc = configIngressRespTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed ingress response table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

    rc = configGangedLinkTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed ganged link table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    // Limerock routing tables
    rc = configRmapTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed Rmap table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

    rc = configRidTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed Rid table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }

    rc = configRlanTable( nodeId, pSwitchInfo );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed Rlan table config for nodeId %d, lwswitch physicalId %d, rc %d.",
                    nodeId, pSwitchInfo->switchphysicalid(), rc);
        return rc;
    }
#endif

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configSwitchPorts( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                     std::set<enum PortType> &portTypes )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::switchPortConfigRequest *pConfigRequest = NULL;
    lwswitch::switchPortInfo *pCfg, *pMsg;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    int portIndex;
    PortKeyType key;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d",
                nodeId, pSwitchInfo->switchphysicalid());

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( portIndex = 0; portIndex < NUM_PORTS_PER_LWSWITCH; portIndex++ )
    {
        key.portIndex = portIndex;
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it != mpGfm->mpParser->portInfo.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // nothing to config
            continue;
        }

        if ( !pCfg )
        {
            continue;
        }

        if ( !pCfg->has_config() || !pCfg->config().has_type() ||
             ( portTypes.find( pCfg->config().type() ) == portTypes.end() ) )
        {
            // only configure the specific port types
            continue;
        }

        if ( !pMessage )
        {
            pMessage = new lwswitch::fmMessage();
            pMessage->set_type( lwswitch::FM_SWITCH_PORT_CONFIG_REQ );
        }

        if ( !pConfigRequest )
        {
            pConfigRequest = new lwswitch::switchPortConfigRequest;
            pMessage->set_allocated_portconfigrequest( pConfigRequest );
            pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
        }

        pMsg = pConfigRequest->add_info();
        pMsg->CopyFrom( *pCfg );
    }

    if ( pMessage )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed switch port config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configSwitchPortsWithTypes( std::set<enum PortType> &portTypes )
{
    FM_ERROR_CODE rc;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;
    SwitchKeyType key;

    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end();
          it++ )
    {
        key = it->first;
        rc = configSwitchPorts( key.nodeId, it->second, portTypes );
        if ( rc != FM_SUCCESS )
        {
            return rc;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configLwswitches( uint32_t nodeId, std::set<enum PortType> &portTypes )
{
    FM_ERROR_CODE rc;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;

    PRINT_DEBUG("%d", "nodeId %d", nodeId);

    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end(); it++ )
    {
        SwitchKeyType key = it->first;
        lwswitch::switchInfo* pSwitchInfo = it->second;
        rc = configOneLwswitch( nodeId, pSwitchInfo, portTypes );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed to config LWSwitch for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, key.physicalId, rc);
            return rc;
        }

        PRINT_INFO("%d %d","LWSwitch for nodeId %d, lwswitch physicalId %d is configured.",
                   nodeId, key.physicalId);
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configGpus( uint32_t nodeId )
{
    FM_ERROR_CODE rc;
    std::map <GpuKeyType, lwswitch::gpuInfo * >::iterator it;
    GpuPhyIdToIndexList gpuList;

    PRINT_DEBUG("%d", "nodeId %d", nodeId);

    for ( it = mpGfm->mpParser->gpuCfg.begin();
          it != mpGfm->mpParser->gpuCfg.end(); it++ )
    {
        GpuKeyType key = it->first;
        if ( key.nodeId == nodeId )
        {
            PRINT_DEBUG("%d %d", "nodeId %d, gpuPhysicalId %d",
                        nodeId, key.physicalId );
            // Note: This is for non-shared fabric mode
            // get GPU physical Id to GPU enumeration index derived based on the LWLink connections.
            uint32_t enumIndex = 0;
            if ( !mpGfm->getGpuEnumIndex( nodeId, key.physicalId, enumIndex ) )
            {
                PRINT_ERROR( "%d", "failed to get GPU enumeration index for PhysicalId %d",
                             key.physicalId );
                // keep enumeration index as physical id itself.
                enumIndex = key.physicalId;
            }
            gpuList.push_back( std::make_pair(key.physicalId, enumIndex) );
        }
    }

    return configGpus( nodeId, ILWALID_FABRIC_PARTITION_ID, gpuList );
}

// Generate FM message to config GPUs, attach or detach GPUs
// on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
lwswitch::fmMessage *
DcgmFabricConfig::generateConfigGpusMsg( uint32_t nodeId, uint32_t partitionId,
                                         GpuPhyIdToIndexList &gpuList,
                                         lwswitch::FabricManagerMessageType configGPUMsgType )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::gpuConfigRequest *pConfigRequest = NULL;
    lwswitch::gpuAttachRequest *pActivateRequest = NULL;
    lwswitch::gpuDetachRequest *pDeactivateRequest = NULL;
    lwswitch::gpuInfo *pCfg, *pMsg;
    std::map <GpuKeyType, lwswitch::gpuInfo * >::iterator it;
    GpuKeyType gpuKey;
    uint32_t gpuCount = 0;
    char uuid[DCGM_DEVICE_UUID_BUFFER_SIZE];

    if ( ( configGPUMsgType != lwswitch::FM_GPU_CONFIG_REQ ) &&
         ( configGPUMsgType != lwswitch::FM_GPU_ATTACH_REQ ) &&
         ( configGPUMsgType != lwswitch::FM_GPU_DETACH_REQ ) )
    {
        PRINT_ERROR("%d", "Invalid GPU config message type configGPUMsgType %d",
                    configGPUMsgType);
        return pMessage;
    }

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type(configGPUMsgType);
    gpuKey.nodeId = nodeId;

    GpuPhyIdToIndexList::iterator git;
    for ( git = gpuList.begin(); git != gpuList.end(); git++ )
    {
        GpuPhyIdToIndexPair gpuPhyIdToIndex = (*git);
        gpuKey.physicalId = gpuPhyIdToIndex.first;
        uint32_t gpuEnumIndex = gpuPhyIdToIndex.second;
        pMsg = NULL;

        it = mpGfm->mpParser->gpuCfg.find(gpuKey);
        if ( it != mpGfm->mpParser->gpuCfg.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // nothing to config
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d", "Invalid gpuInfo nodeId %d, physicalId %d.",
                        nodeId, gpuKey.physicalId);
            continue;
        }

        pCfg->set_gpuenumindex( gpuEnumIndex );

        switch ( configGPUMsgType )
        {
        case lwswitch::FM_GPU_CONFIG_REQ:
            if ( !pConfigRequest )
            {
                pConfigRequest = new lwswitch::gpuConfigRequest;
                pMessage->set_allocated_gpuconfigreq( pConfigRequest );
                pConfigRequest->set_partitionid( partitionId );
            }
            pMsg = pConfigRequest->add_info();
            break;

        case lwswitch::FM_GPU_ATTACH_REQ:
            if ( !pActivateRequest )
            {
                pActivateRequest = new lwswitch::gpuAttachRequest;
                pMessage->set_allocated_gpuattachreq( pActivateRequest );
                pActivateRequest->set_partitionid( partitionId );
            }
            pMsg = pActivateRequest->add_info();
            break;

        case lwswitch::FM_GPU_DETACH_REQ:
            if ( !pDeactivateRequest )
            {
                pDeactivateRequest = new lwswitch::gpuDetachRequest;
                pMessage->set_allocated_gpudetachreq( pDeactivateRequest );
                pDeactivateRequest->set_partitionid( partitionId );
            }
            pMsg = pDeactivateRequest->add_info();
            break;

        default:
            // Should not come here, as the msg type is checked earlier.
            PRINT_ERROR("%d", "Invalid GPU config message type configGPUMsgType %d",
                        configGPUMsgType);
            break;
        }

        if (pMsg)
        {
            pMsg->CopyFrom( *pCfg );
            if (mpGfm->getGpuUuid(nodeId, gpuKey.physicalId, uuid))
            {
                pMsg->set_uuid( uuid );
            }
            gpuCount++;
        }
    }

    if ( gpuCount == 0 )
    {
        // there is no GPUs in the partition
        delete pMessage;
        pMessage = NULL;
    }

    return pMessage;
}

// config GPUs, GPUs on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
FM_ERROR_CODE
DcgmFabricConfig::configGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList)
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = generateConfigGpusMsg( nodeId, partitionId,
                                                           gpuList,
                                                           lwswitch::FM_GPU_CONFIG_REQ);
    if ( pMessage )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed GPU config for nodeId %d, partitionId %d, rc %d.",
                        nodeId, partitionId, rc);

            lwswitch::fmMessage errMsg = *pMessage;
            if ( partitionId == ILWALID_FABRIC_PARTITION_ID ) {
                handleConfigError( nodeId, ERROR_SOURCE_SW_GLOBALFM,
                                   ERROR_TYPE_CONFIG_GPU_FAILED, errMsg);
            } else {
                handleSharedLWSwitchPartitionConfigError( nodeId, partitionId, ERROR_SOURCE_SW_GLOBALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
        }
    }

    return rc;
}

// attach GPUs on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
FM_ERROR_CODE
DcgmFabricConfig::attachGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList)
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = generateConfigGpusMsg( nodeId, partitionId,
                                                           gpuList,
                                                           lwswitch::FM_GPU_ATTACH_REQ);
    if ( pMessage )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed GPU attach for nodeId %d, partitionId %d, rc %d.",
                        nodeId, partitionId, rc);

            lwswitch::fmMessage errMsg = *pMessage;
            // handle partition config error
            handleSharedLWSwitchPartitionConfigError( nodeId, partitionId, ERROR_SOURCE_SW_GLOBALFM,
                                                      ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
        }
    }

    return rc;
}

// detach GPUs on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
FM_ERROR_CODE
DcgmFabricConfig::detachGpus( uint32_t nodeId, uint32_t partitionId, GpuPhyIdToIndexList &gpuList)
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = generateConfigGpusMsg( nodeId, partitionId,
                                                           gpuList,
                                                           lwswitch::FM_GPU_DETACH_REQ);
    if ( pMessage )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed GPU detach for nodeId %d, partitionId %d, rc %d.",
                        nodeId, partitionId, rc);

            lwswitch::fmMessage errMsg = *pMessage;
            // handle partition config error
            handleSharedLWSwitchPartitionConfigError( nodeId, partitionId, ERROR_SOURCE_SW_GLOBALFM,
                                                      ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
        }
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configIngressReqEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::switchPortRequestTable *pConfigRequest;
    lwswitch::switchPortRequestTableInfo *info = NULL;
    ingressRequestTable *pCfg, *pMsg;
    std::map <ReqTableKeyType, ingressRequestTable * >::iterator it;
    ReqTableKeyType key;
    int count = 0;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_INGRESS_REQUEST_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortRequestTable;
    pMessage->set_allocated_requesttablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < INGRESS_REQ_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->reqEntry.find(key);
        if ( it != mpGfm->mpParser->reqEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // no entry in this index
            // indicating a gap in the index space
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid reqEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );

            // previous index on this port is NULL, starting a new info,
            // and setup the first index for the current contiguous index block of entries
            info->set_firstindex( key.index );
        }
        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );
        removeDisabledPortsFromIngressReqEntry( nodeId, pSwitchInfo->switchphysicalid(), pMsg );

        if ( mpGfm->isSharedFabricMode() )
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        PRINT_DEBUG("%d %d %d %d %d",
                    "portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ingress request config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configIngressRespEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::switchPortResponseTable *pConfigRequest;
    lwswitch::switchPortResponseTableInfo *info = NULL;
    ingressResponseTable *pCfg, *pMsg;
    std::map <RespTableKeyType, ingressResponseTable * >::iterator it;
    RespTableKeyType key;
    int count = 0;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_INGRESS_RESPONSE_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortResponseTable;
    pMessage->set_allocated_responsetablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < INGRESS_RESP_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->respEntry.find(key);
        if ( it != mpGfm->mpParser->respEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // no entry in this index
            // indicating a gap in the index space
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid reqEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );

            // previous index on this port is NULL, starting a new info,
            // and setup the first index for the current contiguous index block of entries
            info->set_firstindex( key.index );
        }

        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );
        removeDisabledPortsFromIngressRespEntry( nodeId, pSwitchInfo->switchphysicalid(), pMsg );

        if ( mpGfm->isSharedFabricMode() )
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        PRINT_DEBUG("%d %d %d %d %d",
                    "portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ingress response config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);
    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configGangedLinkEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::switchPortGangedLinkTable *pConfigRequest;
    lwswitch::switchPortGangedLinkTableInfo *info = NULL;
    int32_t *pCfg, *pMsg;
    std::map <GangedLinkTableKeyType, int32_t * >::iterator it;
    GangedLinkTableKeyType key;
    gangedLinkTable *table = NULL;

    int count = 0;

    if ( mpGfm->mpParser->gangedLinkEntry.size() == 0 )
    {
        return FM_SUCCESS;
    }

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_GANGED_LINK_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortGangedLinkTable;
    pMessage->set_allocated_gangedlinktablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < GANGED_LINK_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->gangedLinkEntry.find(key);
        if ( it != mpGfm->mpParser->gangedLinkEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // nothing to config
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid gangedLinkEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );
            table = new gangedLinkTable;
            info->set_allocated_table( table );
        }

        if ( table ) table->add_data(*pCfg);
        count++;
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ganged link config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);
    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configIngressReqTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configIngressReqEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::sendPeerLFMInfo( uint32_t nodeId )
{
    // send all other Node's ID and IP information to the peer
    // for establishing peer LFM connections
    if (mpGfm->mpParser->NodeCfg.size() < 2)
    {
        // nothing to do if we have only one node, ie no LFM to LFM connection needed
        return FM_SUCCESS;
    }

    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    lwswitch::nodeInfoMsg* nodeInfoMsg = new lwswitch::nodeInfoMsg();
    
    // copy each node information
    for ( it = mpGfm->mpParser->NodeCfg.begin();
          it != mpGfm->mpParser->NodeCfg.end(); it++ )
    {
        lwswitch::nodeInfo* nodeInfo = nodeInfoMsg->add_info();
        pNode = it->second;
        nodeInfo->set_nodeid( pNode->nodeId );
        nodeInfo->set_ipaddress( pNode->IPAddress->c_str() );
    }

    FM_ERROR_CODE rc;
    lwswitch::fmMessage *pMessage = NULL;
    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_NODE_INFO_MSG );
    pMessage->set_allocated_nodeinfomsg( nodeInfoMsg );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d", "Failed to send peer LFM information to nodeId %d \n", nodeId);
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::sendConfigInitDoneReqMsg( uint32_t nodeId )
{
    FM_ERROR_CODE rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::configInitDoneReq *pInitDoneReq;

    // prepare the response message    
    pInitDoneReq = new lwswitch::configInitDoneReq();

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_CONFIG_INIT_DONE_REQ );
    pMessage->set_allocated_initdonereq( pInitDoneReq );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true);
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d", "Failed to send CONFIG_INIT_DONE_REQ to nodeId %d \n", nodeId);
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::sendConfigDeInitReqMsg( uint32_t nodeId )
{
    FM_ERROR_CODE rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::configDeInitReq *pDeInitReq;

    // prepare the response message    
    pDeInitReq = new lwswitch::configDeInitReq();

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_CONFIG_DEINIT_REQ );
    pMessage->set_allocated_deinitreq( pDeInitReq );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d", "Failed to send CONFIG_DEINIT_REQ to nodeId %d \n", nodeId);
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configIngressRespTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configIngressRespEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configGangedLinkTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configGangedLinkEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

// activate or deactivate ingress request tables entries for a
// specified shared lwswitch partition
// ingress request entries are set to valid when the partition is activated
// ingress request entries are set to invalid when the partition is deactivated
FM_ERROR_CODE
DcgmFabricConfig::configIngressReqTable( uint32_t nodeId, uint32_t partitionId,
                                         uint32_t switchPhysicalId, uint32_t portNum,
                                         std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::switchPortRequestTable *pConfigRequest;
    lwswitch::switchPortRequestTableInfo *info = NULL;
    ingressRequestTable *pCfg, *pMsg;
    std::map <ReqTableKeyType, ingressRequestTable * >::iterator it;
    ReqTableKeyType key;
    int count = 0;
    accessPort *accessPortInfo = mpGfm->mpParser->getAccessPortInfo( nodeId, switchPhysicalId, portNum );

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_INGRESS_REQUEST_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortRequestTable;
    pMessage->set_allocated_requesttablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( switchPhysicalId );

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for ( std::list<uint32_t>::iterator it = gpuPhysicalIds.begin();
          it != gpuPhysicalIds.end(); it++ )
    {
        uint32_t gpuPhysicalId = *it;

        // the access port is connected to the GPU directly, do not program this GPU on this port
        if ( accessPortInfo && accessPortInfo->has_farpeerid() &&
             ( accessPortInfo->farpeerid() == gpuPhysicalId ) )
        {
            continue;
        }

        uint32_t gpuEndpointID = nodeId * MAX_NUM_GPUS_PER_NODE + gpuPhysicalId;
        uint32_t ingressReqIndex = gpuEndpointID << 2;

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_firstindex( ingressReqIndex );
        info->set_partitionid( partitionId );

        for ( key.index = ingressReqIndex;
              key.index < ingressReqIndex + NUM_INGR_REQ_ENTRIES_PER_GPU;
              key.index++ )
        {
            std::map <ReqTableKeyType, ingressRequestTable *>::iterator entryIt;
            entryIt = mpGfm->mpParser->reqEntry.find(key);
            if ( entryIt != mpGfm->mpParser->reqEntry.end() )
            {
                pCfg = entryIt->second;
                pMsg = info->add_entry();
                pMsg->CopyFrom( *pCfg );
                removeDisabledPortsFromIngressReqEntry( nodeId, switchPhysicalId, pMsg );

                // modify the valid bit
                pMsg->set_entryvalid( activate ? 1 : 0 );
                count++;

                PRINT_DEBUG("%d %d %d %d %d",
                            "portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                             portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
            }
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%s %d %d %d %d %d",
                         "Failed to %s nodeId %d, lwswitch physicalId %d, portNum %d, count %d, rc %d",
                         activate ? "activate" : "deactivate",
                        nodeId, switchPhysicalId, portNum, count, rc);

            // handle partition config error
            lwswitch::fmMessage errMsg = *pMessage;
            handleSharedLWSwitchPartitionConfigError( nodeId, partitionId, ERROR_SOURCE_SW_GLOBALFM,
                                                      ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

// activate or deactivate ingress response tables entries for a
// specified shared lwswitch partition
// ingress response entries are set to valid when the partition is activated
// ingress response entries are set to invalid when the partition is deactivated
FM_ERROR_CODE
DcgmFabricConfig::configIngressRespTable( uint32_t nodeId, uint32_t partitionId,
                                          uint32_t switchPhysicalId, uint32_t portNum,
                                          std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::switchPortResponseTable *pConfigRequest;
    lwswitch::switchPortResponseTableInfo *info = NULL;
    ingressResponseTable *pCfg, *pMsg;
    std::map <RespTableKeyType, ingressResponseTable * >::iterator it;
    RespTableKeyType key;
    int count = 0;
    accessPort *accessPortInfo = mpGfm->mpParser->getAccessPortInfo( nodeId, switchPhysicalId, portNum );

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_INGRESS_RESPONSE_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortResponseTable;
    pMessage->set_allocated_responsetablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( switchPhysicalId );

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for ( std::list<uint32_t>::iterator it = gpuPhysicalIds.begin();
          it != gpuPhysicalIds.end(); it++ )
    {
        uint32_t gpuPhysicalId = *it;

        // the access port is connected to the GPU directly, do not program this GPU on this port
        if ( accessPortInfo && accessPortInfo->has_farpeerid() &&
             ( accessPortInfo->farpeerid() == gpuPhysicalId ) )
        {
            continue;
        }

        uint32_t gpuEndpointID = nodeId * MAX_NUM_GPUS_PER_NODE + gpuPhysicalId;
        uint32_t ingressRespIndex = gpuEndpointID * NUM_LWLINKS_PER_GPU;

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_firstindex( ingressRespIndex );
        info->set_partitionid( partitionId );

        for ( key.index = ingressRespIndex;
              key.index < ingressRespIndex + NUM_LWLINKS_PER_GPU;
              key.index++ )
        {
            std::map <RespTableKeyType, ingressResponseTable *>::iterator entryIt;
            entryIt = mpGfm->mpParser->respEntry.find(key);
            if ( entryIt != mpGfm->mpParser->respEntry.end() )
            {
                pCfg = entryIt->second;
                pMsg = info->add_entry();
                pMsg->CopyFrom( *pCfg );
                removeDisabledPortsFromIngressRespEntry( nodeId, switchPhysicalId, pMsg );

                // modify the valid bit
                pMsg->set_entryvalid( activate ? 1 : 0 );
                count++;

                PRINT_DEBUG("%d %d %d %d %d",
                            "portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                             portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
            }
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%s %d %d %d %d %d",
                         "Failed to %s nodeId %d, lwswitch physicalId %d, portNum %d, count %d, rc %d",
                         activate ? "activate" : "deactivate",
                        nodeId, switchPhysicalId, portNum, count, rc);

            // handle partition config error
            lwswitch::fmMessage errMsg = *pMessage;
            handleSharedLWSwitchPartitionConfigError( nodeId, partitionId, ERROR_SOURCE_SW_GLOBALFM,
                                                      ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configSharedLWSwitchPartitionRoutingTable( uint32_t nodeId, PartitionInfo &partInfo, bool activate)
{
    FM_ERROR_CODE rc = FM_SUCCESS;

    std::list<uint32_t> gpuPhysicalIds;
    for ( PartitionGpuInfoList::iterator it = partInfo.gpuInfo.begin();
          it != partInfo.gpuInfo.end(); it++ )
    {
        PartitionGpuInfo gpuinfo = *it;
        gpuPhysicalIds.push_back( gpuinfo.physicalId );
    }

    uint32_t partitionId = partInfo.partitionId;

    // program add switches in the partition
    for ( PartitionSwitchInfoList::iterator it = partInfo.switchInfo.begin();
          it != partInfo.switchInfo.end(); it++ )
    {
        PartitionSwitchInfo switchinfo = *it;
        uint32_t switchPhysicalId = switchinfo.physicalId;

        // program routing entries on enabled ports
        for ( uint32_t portNum = 0; portNum < LWLINK_MAX_DEVICE_CONN; portNum++ ) {
            // skip if the link is not enabled
            if ( !(switchinfo.enabledLinkMask & ((uint64)1 << portNum)) )
                continue;

            rc = configIngressReqTable( nodeId, partitionId, switchPhysicalId,
                                        portNum, gpuPhysicalIds, activate );
            if ( rc != FM_SUCCESS )
            {
                PRINT_ERROR("%s %d %d", "Failed to %s partitionId %d with error %d.",
                            activate ? "activate" : "deactivate", partitionId, rc);
                FM_SYSLOG_ERR("Failed to %s partitionId %d.",
                              activate ? "activate" : "deactivate", partitionId);
            }

            rc = configIngressRespTable( nodeId, partitionId, switchPhysicalId,
                                         portNum, gpuPhysicalIds, activate );
            if ( rc != FM_SUCCESS )
            {
                PRINT_ERROR("%s %d %d", "Failed to %s partitionId %d with error %d.",
                            activate ? "activate" : "deactivate", partitionId, rc);
                FM_SYSLOG_ERR("Failed to %s partitionId %d.",
                              activate ? "activate" : "deactivate", partitionId);
            }
        }
    }

    return rc;
}

// configure the GPU fabric address for GPUs in a specified partition
FM_ERROR_CODE
DcgmFabricConfig::configSharedLWSwitchPartitionGPUs( uint32_t nodeId, PartitionInfo &partInfo )
{
    // get the list of GPUs for this partition
    GpuPhyIdToIndexList gpuList;
    PartitionGpuInfoList::iterator it;

    // clear provided gpu list.
    gpuList.clear();
    // go through each GPU and create a list of physical id to enum index pair
    for ( it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++ )
    {
        PartitionGpuInfo& gpuinfo = *it;

        // do not config fabric address on the GPU
        // if the GPU has no LWLink enabled on it
        if ( gpuinfo.numEnabledLinks == 0 )
            continue;

        gpuList.push_back( std::make_pair(gpuinfo.physicalId, gpuinfo.dynamicInfo.gpuIndex) );
    }

    FM_ERROR_CODE rc = configGpus( nodeId, partInfo.partitionId, gpuList ) ;
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d", "Failed to config GPUs for partitionId %d with error %d.",
                    partInfo.partitionId, rc);
        FM_SYSLOG_ERR("Failed to config GPUs for partitionId %d.", partInfo.partitionId);
    }

    return rc;
}

// attach the GPUs in a specified partition
FM_ERROR_CODE
DcgmFabricConfig::configSharedLWSwitchPartitionAttachGPUs( uint32_t nodeId, uint32_t partitionId )
{
    GpuPhyIdToIndexList gpuList;

    // TODO - Attach/Detach GPU don't require the GPU IDs to be passed.
    // This will be addressed in another patch. So filling a dummy
    // GPU information to get us going with existing code flow.
    gpuList.push_back( std::make_pair(1, 1) );

    FM_ERROR_CODE rc = attachGpus( nodeId, partitionId, gpuList ) ;
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d", "Failed to attach GPUs for partitionId %d with error %d.",
                    partitionId, rc);
    }

    // Make sure all attach GPU responses are back
    if ( waitForPartitionConfigCompletion(nodeId, partitionId, PARTITION_CFG_TIMEOUT_MS ) == false)
    {
        PRINT_ERROR("%d", "Attach GPUs for partition id %d has timed out", partitionId);
        rc = FM_CFG_TIMEOUT;
    }

    return rc;
}

// detach the GPUs in a specified partition
FM_ERROR_CODE
DcgmFabricConfig::configSharedLWSwitchPartitionDetachGPUs( uint32_t nodeId, uint32_t partitionId,
                                                           bool inErrHdlr )
{
    GpuPhyIdToIndexList gpuList;

    // TODO - Attach/Detach GPU don't require the GPU IDs to be passed.
    // This will be addressed in another patch. So filling a dummy
    // GPU information to get us going with existing code flow.
    gpuList.push_back( std::make_pair(1, 1) );

    FM_ERROR_CODE rc = detachGpus( nodeId, partitionId, gpuList ) ;
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d", "Failed to detach GPUs for partitionId %d with error %d.",
                    partitionId, rc);
    }

    if ( inErrHdlr )
    {
        // no need to wait for all responses to come back, as in the case of error handling
        return rc;
    }

    // Make sure all detach GPU responses are back
    if ( waitForPartitionConfigCompletion(nodeId, partitionId, PARTITION_CFG_TIMEOUT_MS ) == false)
    {
        PRINT_ERROR("%d", "Detach GPUs for partition id %d has timed out", partitionId);
        rc = FM_CFG_TIMEOUT;
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::sendSwitchDisableLinkReq( uint32_t nodeId, uint32_t physicalId, uint64 disableMask)
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::switchDisableLinkRequest *pDisableReqMsg = new lwswitch::switchDisableLinkRequest();

    // prepare the request message    
    pDisableReqMsg->set_switchphysicalid( physicalId );
    for ( uint32_t port = 0; port < NUM_PORTS_PER_LWSWITCH; port++ )
    {
        if ( (disableMask & ((uint64_t)1 << port)) )
        {
            pDisableReqMsg->add_portnum( port );
        }
    }

    pMessage->set_type( lwswitch::FM_SWITCH_DISABLE_LINK_REQ );
    pMessage->set_allocated_switchdisablelinkreq( pDisableReqMsg );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true);
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR("%d", "Failed to send FM_SWITCH_DISABLE_LINK_REQ to nodeId %d \n", nodeId);
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configDisableSwitchLinks( uint32_t nodeId, uint64 disableMask)
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;

    // find the LWSwitch associated with specified node
    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end(); it++ )
    {
        SwitchKeyType key = it->first;
        if ( key.nodeId == nodeId )
        {
            uint32_t physicalId = key.physicalId;
            rc = sendSwitchDisableLinkReq ( nodeId, physicalId, disableMask);
            if ( rc != FM_SUCCESS )
            {
                PRINT_ERROR( "%d", "configDisableSwitchTrunkLinks failed for nodeId %d", nodeId );
                return rc;
            }
        }
    }

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::SendMessageToLfm( uint32_t nodeId, uint32_t partitionId,
                                    lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    dcgmReturn_t ret;
    FM_ERROR_CODE rc = FM_SUCCESS;

    if ( !pFmMessage )
    {
        PRINT_DEBUG("%d %d","Invalid message to nodeId %d, partitionId %d.",
                    nodeId, partitionId);
        return FM_MSG_SEND_ERR;
    }

    // add request to our context for tracking
    // before sending the message, add it to the list as the response can
    // come before even we add it to the list.
    int id = mpGfm->getControlMessageRequestId(nodeId);
    pFmMessage->set_requestid(id);
    if (trackReq) {
        PRINT_DEBUG("%d %d", "addPendingConfigRequest RequestId=%d type=%d", id, pFmMessage->type());
        addPendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    }

    ret = mpGfm->SendMessageToLfm(nodeId, pFmMessage, trackReq);
    if ( ret != DCGM_ST_OK )
    {
        PRINT_ERROR("%d %d %d","Failed to send to nodeId %d, partitionId %d, ret %d",
                    nodeId, partitionId, ret);
        rc = FM_MSG_SEND_ERR;

        if ( trackReq )
            removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    }

    delete pFmMessage;
    return rc;
}

void
DcgmFabricConfig::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    PRINT_DEBUG("%d", "message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_NODE_GLOBAL_CONFIG_RSP:
        dumpMessage(pFmMessage);
        handleNodeGlobalConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_SWITCH_PORT_CONFIG_RSP:
        dumpMessage(pFmMessage);
        handleSWPortConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_INGRESS_REQUEST_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleIngReqTblConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_INGRESS_RESPONSE_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleIngRespTblConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_GANGED_LINK_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleGangedLinkTblConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_GPU_CONFIG_RSP:
        dumpMessage(pFmMessage);
        handleGpuConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_GPU_ATTACH_RSP:
        dumpMessage(pFmMessage);
        handleGpuAttachRespMsg( pFmMessage );
        break;

    case lwswitch::FM_GPU_DETACH_RSP:
        dumpMessage(pFmMessage);
        handleGpuDetachRespMsg( pFmMessage );
        break;

    case lwswitch::FM_CONFIG_INIT_DONE_RSP:
        dumpMessage(pFmMessage);
        handleConfigInitDoneRespMsg( pFmMessage );
        break;

    case lwswitch::FM_CONFIG_DEINIT_RSP:
        dumpMessage(pFmMessage);
        handleConfigDeinitRespMsg( pFmMessage );
        break;

    case lwswitch::FM_SWITCH_DISABLE_LINK_RSP:
        dumpMessage(pFmMessage);
        handleConfigSwitchDisableLinkRespMsg( pFmMessage );
        break;

    case lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_RSP:
        dumpMessage(pFmMessage);
        handleConfigGpuSetDisabledLinkMaskRespMsg( pFmMessage );
        break;
    case lwswitch::FM_NODE_INFO_ACK:
        dumpMessage(pFmMessage);
        handleConfigNodeInfoAckMsg( pFmMessage );
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    case lwswitch::FM_RMAP_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRmapTableConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_RID_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRidTableConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_RLAN_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRlanTableConfigRespMsg( pFmMessage );
        break;
#endif

    default:
        PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
        break;
    }
}

void
DcgmFabricConfig::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    PRINT_DEBUG("%s", "%s", msgText.c_str());
#endif
}

/**
 *  on GFM, handle FM_NODE_GLOBAL_CONFIG_RSP message from LFM
 */
void
DcgmFabricConfig::handleNodeGlobalConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_globalconfigresponse() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::nodeGlobalConfigResponse respMsg = pFmMessage->globalconfigresponse();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        PRINT_ERROR("%d %d", "FM_NODE_GLOBAL_CONFIG_RSP got error %d from node ID %d.",
                    respMsg.status(), nodeId);

        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_NODE_FAILED, errMsg );
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
}

/**
 *  on GFM, handle FM_SWITCH_PORT_CONFIG_RSP message from LFM
 */
void
DcgmFabricConfig::handleSWPortConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_portconfigresponse() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::switchPortConfigResponse &respMsg = pFmMessage->portconfigresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d", "FM_SWITCH_PORT_CONFIG_RSP got error %d from node ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

/**
 *  on GFM, handle FM_INGRESS_REQUEST_TABLE_RSP message from LFM
 */
void
DcgmFabricConfig::handleIngReqTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_requesttableresponse() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::switchPortRequestTableResponse &respMsg = pFmMessage->requesttableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d %d",
                        "FM_INGRESS_REQUEST_TABLE_RSP got error %d from node ID %d, partition ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

/**
 *  on GFM, handle FM_INGRESS_RESPONSE_TABLE_RSP message from LFM
 */
void
DcgmFabricConfig::handleIngRespTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_responsetableresponse() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::switchPortResponseTableResponse &respMsg = pFmMessage->responsetableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d %d",
                        "FM_INGRESS_RESPONSE_TABLE_RSP got error %d from node ID %d, partition ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

/**
 *  on GFM, handle FM_GANGED_LINK_TABLE_RSP message from LFM
 */
void
DcgmFabricConfig::handleGangedLinkTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gangedlinktableresponse() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::switchPortGangedLinkTableResponse &respMsg = pFmMessage->gangedlinktableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d", "FM_GANGED_LINK_TABLE_RSP got error %d from node ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

/**
 *  on GFM, handle FM_GPU_CONFIG_RSP message from LFM
 */
void
DcgmFabricConfig::handleGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gpuconfigrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::gpuConfigResponse &respMsg = pFmMessage->gpuconfigrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {

            PRINT_ERROR("%d %d %d %d",
                        "FM_GPU_CONFIG_RSP got error %d from node ID %d, Partition ID %d, GPU ID %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED,
                                                          errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

/**
 *  on GFM, handle FM_GPU_ATTACH_RSP message from LFM
 */
void
DcgmFabricConfig::handleGpuAttachRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gpuattachrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::gpuAttachResponse &respMsg = pFmMessage->gpuattachrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d",
                        "FM_GPU_ATTACH_RSP got error %d from node ID %d, Partition ID %d, GPU ID %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

void
DcgmFabricConfig::handleGpuDetachRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gpudetachrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::gpuDetachResponse &respMsg = pFmMessage->gpudetachrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d",
                        "FM_GPU_DETACH_RSP got error %d from node ID %d, Partition ID %d, GPU ID %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}
/**
 *  on GFM, handle FM_CONFIG_INIT_DONE_RSP message from LFM
 */
void
DcgmFabricConfig::handleConfigInitDoneRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_initdonersp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::configInitDoneRsp &respMsg = pFmMessage->initdonersp();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        PRINT_ERROR("%d %d", "FM_CONFIG_INIT_DONE_RSP got error %d from node ID %d.",
                    respMsg.status(), nodeId);

        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_NODE_FAILED, errMsg );
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
}
/**
 *  on GFM, handle FM_NODE_INFO_ACK message from LFM
 */
void
DcgmFabricConfig::handleConfigNodeInfoAckMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_nodeinfoack() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::nodeInfoAck &respMsg = pFmMessage->nodeinfoack();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        PRINT_ERROR("%d %d", "FM_NODE_INFO_ACK got error %d from node ID %d.",
                    respMsg.status(), nodeId);

        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_NODE_FAILED, errMsg );
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
}


/**
 *  on GFM, handle FM_CONFIG_DEINIT_RSP message from LFM
 */
void
DcgmFabricConfig::handleConfigDeinitRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_deinitrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::configDeInitRsp &respMsg = pFmMessage->deinitrsp();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        PRINT_ERROR("%d %d", "FM_CONFIG_INIT_DONE_RSP got error %d from node ID %d.",
                    respMsg.status(), nodeId);
        // for this deinit failure, don't generate our usual handleConfigError() path
        // as that will generate another config deinit request and potentially this
        // pattern will repeat.
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
}

/**
 *  on GFM, handle FM_SWITCH_DISABLE_LINK_RSP message from LFM
 */
void
DcgmFabricConfig::handleConfigSwitchDisableLinkRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_switchdisablelinkrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::switchDisableLinkResponse &respMsg = pFmMessage->switchdisablelinkrsp();
    uint32_t nodeId = pFmMessage->nodeid();

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        PRINT_ERROR("%d %d", "FM_SWITCH_DISABLE_LINK_RSP got error %d from node ID %d.",
                    respMsg.status(), nodeId);

        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_FAILED, errMsg );
    }

    PRINT_ERROR("%d %d", "FM_SWITCH_DISABLE_LINK_RSP node ID %d, requestId %d.",
                nodeId, pFmMessage->requestid());
    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
}

/**
 *  on GFM, handle FM_GPU_SET_DISABLED_LINK_MASK_RSP message from LFM
 */
void
DcgmFabricConfig::handleConfigGpuSetDisabledLinkMaskRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gpusetdisabledlinkmaskrsp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::gpuSetDisabledLinkMaskResponse &respMsg = pFmMessage->gpusetdisabledlinkmaskrsp();
    uint32_t nodeId = pFmMessage->nodeid();

    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;
    if ( respMsg.has_partitionid() )
    {
        partitionId = respMsg.partitionid();
    }

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        // failure will have corresponding GPU uuid information
        std::string strUuid = "";
        if ( respMsg.has_uuid() )
        {
            strUuid = respMsg.uuid();
        }

        PRINT_ERROR( "%d %d %d %s",
                    "FM_GPU_SET_DISABLED_LINK_MASK_RSP got error %d from node ID %d partidionId %d for GPU Uuid %s",
                    respMsg.status(), nodeId, partitionId, strUuid.c_str() );



        lwswitch::fmMessage errMsg = *pFmMessage;
        if ( partitionId == ILWALID_FABRIC_PARTITION_ID )
        {
            handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
        }
        else
        {
            // handle partition config error
            handleSharedLWSwitchPartitionConfigError( nodeId, partitionId,
                                                      ERROR_SORUCE_SW_LOCALFM,
                                                      ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

void
DcgmFabricConfig::handleConfigError( uint32_t nodeId, GlobalFMErrorSource errSource,
                                     GlobalFMErrorTypes errType, lwswitch::fmMessage &errMsg )
{
    DcgmGlobalFMErrorHndlr errHndlr(mpGfm, nodeId, 0, errSource, errType, errMsg);
    errHndlr.processErrorMsg();
}

void
DcgmFabricConfig::handleSharedLWSwitchPartitionConfigError( uint32_t nodeId, uint32_t partitionId,
                                                            GlobalFMErrorSource errSource,
                                                            GlobalFMErrorTypes errType,
                                                            lwswitch::fmMessage &errMsg )
{
    DcgmGlobalFMErrorHndlr errHndlr(mpGfm, nodeId, partitionId, errSource, errType, errMsg);
    errHndlr.processErrorMsg();
}

void
DcgmFabricConfig::addPendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                           uint32_t partitionId  )
{
    DcgmFMAutoLock lock(mLock);

    std::map <ConfigRequestKeyType, unsigned int>::iterator it;
    ConfigRequestKeyType key;

    key.nodeId = nodeId;
    key.requestId = requestId;
    key.partitionId = partitionId;

    it = mPendingConfigReqMap.find( key );
    if ( it != mPendingConfigReqMap.end() )
    {
        PRINT_DEBUG("%d %d %d","Request nodeId %d, partitionId %d, requestId %d is already pending",
                    nodeId, partitionId, requestId);
        return;
    }

    mPendingConfigReqMap.insert(make_pair(key, requestId));
}

void
DcgmFabricConfig::removePendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                              uint32_t partitionId )
{
    DcgmFMAutoLock lock(mLock);

    std::map <ConfigRequestKeyType, unsigned int>::iterator it;
    ConfigRequestKeyType key;

    key.nodeId = nodeId;
    key.requestId = requestId;
    key.partitionId = partitionId;

    it = mPendingConfigReqMap.find( key );
    if ( it == mPendingConfigReqMap.end() )
    {
        PRINT_DEBUG("%d %d %d","Request nodeId %d, partitionId %d, requestId %d is not pending",
                    nodeId, partitionId, requestId);
        return;
    }

    mPendingConfigReqMap.erase( key );
}

void
DcgmFabricConfig::clearPartitionPendingConfigRequest( uint32_t nodeId, uint32_t partitionId )
{
    DcgmFMAutoLock lock(mLock);

    std::map <ConfigRequestKeyType, unsigned int>::iterator it;
    ConfigRequestKeyType key;

    it = mPendingConfigReqMap.begin();
    while ( it != mPendingConfigReqMap.end() )
    {
        key = it->first;
        it++;
        if ( ( key.nodeId == nodeId ) && ( key.partitionId = partitionId ) )
        {
            mPendingConfigReqMap.erase( key );
        }
    }
}

bool DcgmFabricConfig::isPendingConfigReqEmpty( uint32_t nodeId, uint32_t partitionId )
{
    DcgmFMAutoLock lock(mLock);
    std::map <ConfigRequestKeyType, unsigned int>::iterator it;

    for ( it = mPendingConfigReqMap.begin();
          it != mPendingConfigReqMap.end();
          it++ )
    {
        ConfigRequestKeyType key = it->first;
        if ( ( key.nodeId == nodeId ) && ( key.partitionId == partitionId ) )
        {
            return false;
        }
    }

    return true;
}

void DcgmFabricConfig::dumpPendingConfigReq( uint32_t nodeId, uint32_t partitionId )
{
    DcgmFMAutoLock lock(mLock);
    std::map <ConfigRequestKeyType, unsigned int>::iterator it;

    PRINT_DEBUG("%d", "Pending Config requests on nodeId %d", nodeId);
    for ( it = mPendingConfigReqMap.begin();
          it != mPendingConfigReqMap.end();
          it++ )
    {
        ConfigRequestKeyType key = it->first;
        if ( ( key.nodeId != nodeId ) || ( key.partitionId != partitionId ) )
             continue;

        PRINT_DEBUG("%d", "requestId: %d", it->second);
    }
}


/**
 *  on GFM, send routing and GPU configuration to LFM when a partition is activated or deactivated
 */
FM_ERROR_CODE
DcgmFabricConfig::configSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                                 bool activate, bool inErrHdlr )
{
    FM_ERROR_CODE rc;

    // setup ingress request and response table
    rc = configSharedLWSwitchPartitionRoutingTable( nodeId, partInfo, activate );
    if ( rc != FM_SUCCESS )
    {
        return rc;
    }

    // configure the GPUs only when the partition is activated
    if ( activate )
    {
        rc = configSharedLWSwitchPartitionGPUs( nodeId, partInfo );
        if ( rc != FM_SUCCESS )
        {
            return rc;
        }
    }

    if ( inErrHdlr )
    {
        // no need to wait for all responses to come back, as in the case of error handling
        return rc;
    }

    // Make sure all config responses are back
    if ( waitForPartitionConfigCompletion(nodeId, partInfo.partitionId, PARTITION_CFG_TIMEOUT_MS) == false )
    {
        PRINT_ERROR("%d", "Config lwswitch for partition id %d has timed out", partInfo.partitionId);
        rc = FM_CFG_TIMEOUT;
    }

    return rc;
}

/**
 *  on GFM, send routing and GPU configurations to LFM when a partition is activated
 */
FM_ERROR_CODE
DcgmFabricConfig::configActivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo )
{
    return configSharedLWSwitchPartition( nodeId, partInfo, true );
}

/**
 *  on GFM, send routing and GPU configurations to LFM when a partition is deactivated
 */
FM_ERROR_CODE
DcgmFabricConfig::configDeactivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                                           bool inErrHdlr )
{
    return configSharedLWSwitchPartition( nodeId, partInfo, false, inErrHdlr );
}

/**
 *  on GFM, send information to LFM about GPU LWLinks to be disabled for a partition
 */
FM_ERROR_CODE
DcgmFabricConfig::configSetGpuDisabledLinkMaskForPartition( uint32_t nodeId, PartitionInfo &partInfo )
{
    FM_ERROR_CODE rc;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuSetDisabledLinkMaskRequest *pDisabledLinkReqMsg = new lwswitch::gpuSetDisabledLinkMaskRequest();
    pDisabledLinkReqMsg->set_partitionid(partInfo.partitionId);

    // go through each GPU and create a list of physical id to enum index pair
    PartitionGpuInfoList::iterator it;
    for ( it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++ )
    {
        PartitionGpuInfo gpuInfo = *it;
        uint64 disabledLinkMask = 0;
        // compute the links to be disabled for the GPU.
        for ( uint32 idx = 0; idx < DCGM_LWLINK_MAX_LINKS_PER_GPU; idx++ )
        {
            if ( !(gpuInfo.enabledLinkMask & ((uint64)1 << idx)) )
            {
                // this link index needs to be disabled for the partition
                disabledLinkMask |= LWBIT64(idx);
            }
        }

        // fill the disable link GPB message.
        lwswitch::gpuDisabledLinkMaskInfoMsg *pMaskInfoMsg = pDisabledLinkReqMsg->add_gpuinfo();
        pMaskInfoMsg->set_uuid( gpuInfo.uuid );
        pMaskInfoMsg->set_disablemask( disabledLinkMask );
    }

    // send the final messages to LFM
    pMessage->set_type( lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_REQ );
    pMessage->set_allocated_gpusetdisabledlinkmaskreq( pDisabledLinkReqMsg );

    rc = SendMessageToLfm( nodeId, partInfo.partitionId, pMessage, true);
    if ( rc != FM_SUCCESS )
    {
        PRINT_ERROR( "%d", "Failed to send FM_GPU_DISABLE_LINK_REQ to nodeId %d \n", nodeId );
        return rc;
    }

    // Wait for the command response from LocalFM
    if ( waitForPartitionConfigCompletion(nodeId, partInfo.partitionId, PARTITION_CFG_TIMEOUT_MS) == false )
    {
        PRINT_ERROR("%d", "SetGpuDisabledLinkMask for partition id %d has timed out", partInfo.partitionId);
        rc = FM_CFG_TIMEOUT;
    }

    return rc;
}

/*
 * Evaluate the switch enabled link mask, remove disabled ports from egress port map, return
 * entry valid
 *
 * The port map could be from ingress request or response table entry.
 */
void
DcgmFabricConfig::removeDisabledPortsFromRoutingEntry( uint32_t nodeId, uint32_t physicalId,
                                                       int32_t *vcmodevalid7_0, int32_t *vcmodevalid15_8,
                                                       int32_t *vcmodevalid17_16, int32_t *entryValid )
{
    uint64_t enabledLinkMask = 0;
    DcgmFMLWSwitchInfoMap::iterator it;
    DcgmFMLWSwitchInfoMap switchInfoMap = mpGfm->getLwSwitchInfoMap();

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        DcgmFMLWSwitchInfoList switchList = it->second;
        DcgmFMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            DcgmFMLWSwitchInfo switchInfo = (*jit);
            if ( switchInfo.physicalId == physicalId )
            {
                enabledLinkMask = switchInfo.enabledLinkMask;
            }
        }
    }

    if ( enabledLinkMask )
    {
        int portNum;

        if ( vcmodevalid7_0 )
        {
            for ( portNum = 0; portNum <= 7;  portNum++ )
            {
                if ( ( enabledLinkMask & ( (uint64_t)1 << portNum ) ) != 0 )
                    continue;

                *vcmodevalid7_0 &= ~( 1 << (4*(portNum)) );
            }
        }

        if ( vcmodevalid15_8 )
        {
            for ( portNum = 8; portNum <= 15;  portNum++ )
            {
                if ( ( enabledLinkMask & ( (uint64_t)1 << portNum ) ) != 0 )
                    continue;

                *vcmodevalid15_8 &= ~( 1 << (4*(portNum - 8)) );
            }
        }

        if ( vcmodevalid17_16 )
        {
            for ( portNum = 16; portNum <= 17;  portNum++ )
            {
                if ( ( enabledLinkMask & ( (uint64_t)1 << portNum ) ) != 0 )
                    continue;

                *vcmodevalid17_16 &= ~( 1 << (4*(portNum - 16)) );
            }
        }

        // modify the entry valid accordingly
        if ( entryValid )
        {
            *entryValid = ( vcmodevalid7_0 && ( *vcmodevalid7_0 != 0 ) ) ||
                          ( vcmodevalid15_8 && ( *vcmodevalid15_8 != 0 ) ) ||
                          ( vcmodevalid17_16 && ( *vcmodevalid17_16 != 0) );
        }
    }
}

void
DcgmFabricConfig::removeDisabledPortsFromIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                                          ingressRequestTable *entry )
{
    if ( !entry ) return;

    // remove disabled ports from egress port map
    int32_t vcmodevalid7_0, vcmodevalid15_8, vcmodevalid17_16, entryValid;

    vcmodevalid7_0 = entry->has_vcmodevalid7_0() ? entry->vcmodevalid7_0() : 0;
    vcmodevalid15_8 = entry->has_vcmodevalid15_8() ? entry->vcmodevalid15_8() : 0;
    vcmodevalid17_16 = entry->has_vcmodevalid17_16() ? entry->vcmodevalid17_16() : 0;
    entryValid = entry->has_entryvalid() ? entry->entryvalid() : 0;

    removeDisabledPortsFromRoutingEntry( nodeId, physicalId,
                                         &vcmodevalid7_0, &vcmodevalid15_8,
                                         &vcmodevalid17_16, &entryValid);

    if ( entry->has_vcmodevalid7_0() ) entry->set_vcmodevalid7_0( vcmodevalid7_0 );
    if ( entry->has_vcmodevalid15_8() ) entry->set_vcmodevalid15_8( vcmodevalid15_8 );
    if ( entry->has_vcmodevalid17_16() ) entry->set_vcmodevalid17_16( vcmodevalid17_16 );
    if ( entry->has_entryvalid() ) entry->set_entryvalid( entryValid );
}

void
DcgmFabricConfig::removeDisabledPortsFromIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                                           ingressResponseTable *entry )
{
    if ( !entry ) return;

    // remove disabled ports from egress port map
    int vcmodevalid7_0, vcmodevalid15_8, vcmodevalid17_16, entryValid;

    vcmodevalid7_0 = entry->has_vcmodevalid7_0() ? entry->vcmodevalid7_0() : 0;
    vcmodevalid15_8 = entry->has_vcmodevalid15_8() ? entry->vcmodevalid15_8() : 0;
    vcmodevalid17_16 = entry->has_vcmodevalid17_16() ? entry->vcmodevalid17_16() : 0;
    entryValid = entry->has_entryvalid() ? entry->entryvalid() : 0;

    removeDisabledPortsFromRoutingEntry( nodeId, physicalId,
                                         &vcmodevalid7_0, &vcmodevalid15_8,
                                         &vcmodevalid17_16, &entryValid);

    if ( entry->has_vcmodevalid7_0() ) entry->set_vcmodevalid7_0( vcmodevalid7_0 );
    if ( entry->has_vcmodevalid15_8() ) entry->set_vcmodevalid15_8( vcmodevalid15_8 );
    if ( entry->has_vcmodevalid17_16() ) entry->set_vcmodevalid17_16( vcmodevalid17_16 );
    if ( entry->has_entryvalid() ) entry->set_entryvalid( entryValid );
}

// return
// true:  when all partition config responses have come back before timeout
// false: when not all partition config responses have come back before timeout
//        or error response has returned.
bool
DcgmFabricConfig::waitForPartitionConfigCompletion( uint32_t nodeId,
                                                    uint32_t partitionId,
                                                    uint32_t timeoutMs )
{
    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
    const unsigned int WAIT_MS = 50; // wait interval

    while (true) {
        if (mpGfm->mGfmPartitionMgr->isPartitionConfigFailed(nodeId, partitionId))
        {
            // no need to wait, as there are errors on the partition already
            return false;
        }

        if ( isPendingConfigReqEmpty( nodeId, partitionId ) )
        {
            // no more pending requests
            return true;
        }

        timeNow = timelib_usecSince1970();
        if ( (timeNow - timeStart) + WAIT_MS*1000 > timeoutMs*1000 ) {
             // elapsed all the time and there are still pending requests.
             break;
        }
        // wait for more time
        usleep( WAIT_MS * 1000 );
    }

    // timed out
    PRINT_ERROR("%d %d", "Timeout for nodeId %d, partitionId %d", nodeId, partitionId);

    // Clear pending request before starting config this partition
    clearPartitionPendingConfigRequest( nodeId, partitionId );
    return false;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)

/*
 * Evaluate the switch enabled link mask, remove disabled ports from rid port list.
 * If port list is empty, the entry is set to invalid
 */
void
DcgmFabricConfig::removeDisabledPortsFromRidEntry( uint32_t nodeId, uint32_t physicalId,
                                                   ridRouteEntry *entry )
{
    // Limerock TODO
    return;
}

/*
 * Evaluate the switch enabled link mask, remove disabled ports from rlan group list.
 * If group list is empty, the entry is set to invalid
 */
void
DcgmFabricConfig::removeDisabledPortsFromRlanEntry( uint32_t nodeId, uint32_t physicalId,
                                                    rlanRouteEntry *entry )
{
    // Limerock TODO
    return;
}

FM_ERROR_CODE
DcgmFabricConfig::configRmapEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRmapTableRequest *pConfigRequest;
    lwswitch::portRmapTableInfo *info = NULL;
    rmapPolicyEntry *pCfg, *pMsg;
    std::map <RmapTableKeyType, rmapPolicyEntry * >::iterator it;
    RmapTableKeyType key;
    int count = 0;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RMAP_TABLE_REQ );

    pConfigRequest = new lwswitch::portRmapTableRequest;
    pMessage->set_allocated_rmaptablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < LR_INGRESS_RMAP_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->rmapEntry.find(key);
        if ( it != mpGfm->mpParser->rmapEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // no entry in this index
            // indicating a gap in the index space
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid rmapEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );

            // previous index on this port is NULL, starting a new info,
            // and setup the first index for the current contiguous index block of entries
            info->set_firstindex( key.index );
        }
        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );

        if ( mpGfm->isSharedFabricMode() )
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        PRINT_DEBUG("%d %d %d %d %d",
                    "portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ingress rmap config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configRidEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRidTableRequest *pConfigRequest;
    lwswitch::portRidTableInfo *info = NULL;
    ridRouteEntry *pCfg, *pMsg;
    std::map <RidTableKeyType, ridRouteEntry * >::iterator it;
    RidTableKeyType key;
    int count = 0;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RID_TABLE_REQ );

    pConfigRequest = new lwswitch::portRidTableRequest;
    pMessage->set_allocated_ridtablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < LR_INGRESS_RID_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->ridEntry.find(key);
        if ( it != mpGfm->mpParser->ridEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // no entry in this index
            // indicating a gap in the index space
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid ridEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );

            // previous index on this port is NULL, starting a new info,
            // and setup the first index for the current contiguous index block of entries
            info->set_firstindex( key.index );
        }
        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );
        removeDisabledPortsFromRidEntry( nodeId, pSwitchInfo->switchphysicalid(), pMsg );

        if ( mpGfm->isSharedFabricMode() )
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_valid( 0 );
        }
        count++;

        PRINT_DEBUG("%d %d %d %d %d",
                    "portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ingress rid config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configRlanEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FM_ERROR_CODE rc = FM_SUCCESS;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRlanTableRequest *pConfigRequest;
    lwswitch::portRlanTableInfo *info = NULL;
    rlanRouteEntry *pCfg, *pMsg;
    std::map <RlanTableKeyType, rlanRouteEntry * >::iterator it;
    RlanTableKeyType key;
    int count = 0;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RLAN_TABLE_REQ );

    pConfigRequest = new lwswitch::portRlanTableRequest;
    pMessage->set_allocated_rlantablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < LR_INGRESS_RLAN_TABLE_SIZE; key.index++ )
    {
        it = mpGfm->mpParser->rlanEntry.find(key);
        if ( it != mpGfm->mpParser->rlanEntry.end() )
        {
            pCfg = it->second;
        }
        else
        {
            // no entry in this index
            // indicating a gap in the index space
            info = NULL;
            continue;
        }

        if ( !pCfg )
        {
            PRINT_ERROR("%d %d %d %d", "Invalid ridEntry nodeId %d, lwswitch physicalId %d, portIndex %d, index %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), portIndex, key.index);
            continue;
        }

        if ( info == NULL )
        {
            info = pConfigRequest->add_info();
            info->set_switchphysicalid( pSwitchInfo->switchphysicalid() );
            info->set_port( portIndex );

            // previous index on this port is NULL, starting a new info,
            // and setup the first index for the current contiguous index block of entries
            info->set_firstindex( key.index );
        }
        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );
        removeDisabledPortsFromRlanEntry( nodeId, pSwitchInfo->switchphysicalid(), pMsg );

        if ( mpGfm->isSharedFabricMode() )
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_valid( 0 );
        }
        count++;

        PRINT_DEBUG("%d %d %d %d %d",
                    "portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d",
                        "Failed ingress rlan config for nodeId %d, lwswitch physicalId %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    PRINT_DEBUG("%d %d %d %d %d",
                "nodeId %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FM_ERROR_CODE
DcgmFabricConfig::configRmapTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRmapEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configRidTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRidEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricConfig::configRlanTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FM_ERROR_CODE rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  NUM_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRlanEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d",
                        "Failed on port nodeId %d, lwswitch physicalId %d, portIndex %d, rc %d.",
                        nodeId, pSwitchInfo->switchphysicalid(), key.portIndex, rc);
            return rc;
        }
    }

    return FM_SUCCESS;
}

void
DcgmFabricConfig::handleRmapTableConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_rmaptablersp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::portRmapTableResponse &respMsg = pFmMessage->rmaptablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d %d",
                        "FM_RMAP_REQUEST_TABLE_RSP got error %d from node ID %d, partition ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

void
DcgmFabricConfig::handleRidTableConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_ridtablersp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::portRidTableResponse &respMsg = pFmMessage->ridtablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d %d",
                        "FM_RID_REQUEST_TABLE_RSP got error %d from node ID %d, partition ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

void
DcgmFabricConfig::handleRlanTableConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_rlantablersp() )
    {
        // empty response
        PRINT_DEBUG("", "Empty response.");
        return;
    }

    const lwswitch::portRlanTableResponse &respMsg = pFmMessage->rlantablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        PRINT_DEBUG("", "No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return;
    }

    for ( int i = 0; i < respMsg.response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg.response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            PRINT_ERROR("%d %d %d %d %d",
                        "FM_RLAN_REQUEST_TABLE_RSP got error %d from node ID %d, partition ID %d, switch ID %d, port %d.",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                handleSharedLWSwitchPartitionConfigError( nodeId, instanceResponse.partitionid(),
                                                          ERROR_SORUCE_SW_LOCALFM,
                                                          ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED, errMsg);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
}

#endif

