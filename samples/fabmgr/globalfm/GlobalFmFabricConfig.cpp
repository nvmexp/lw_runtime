/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>

#include <google/protobuf/text_format.h>
#include "fm_log.h"
#include "GlobalFmFabricConfig.h"
#include "GlobalFmFabricParser.h"
#include "GlobalFabricManager.h"
#include "GlobalFmDegradedModeMgr.h"
#include "FMAutoLock.h"
#include "FMDeviceProperty.h"

FMFabricConfig::FMFabricConfig(GlobalFabricManager *pGfm)
{
    mpGfm = pGfm;
    mCfgMsgTimeoutSec = pGfm->isSimMode() ? FM_CONFIG_MSG_TIMEOUT_SIM : FM_CONFIG_MSG_TIMEOUT;

    // lock is required as the pending config request map will be accessed from
    // request message send, and response message handling
    lwosInitializeCriticalSection( &mLock );
};

FMFabricConfig::~FMFabricConfig()
{
    lwosDeleteCriticalSection( &mLock );
};

void
FMFabricConfig::handleEvent( FabricManagerCommEventType eventType, uint32_t nodeId )
{
    switch (eventType) {
        case FM_EVENT_PEER_FM_CONNECT: {
            // There is no peer FM connection for config, not applicable.
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d FM_EVENT_PEER_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_PEER_FM_DISCONNECT: {
            // There is no peer FM connection for config, not applicable.
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d FM_EVENT_PEER_FM_DISCONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_CONNECT: {
            // configs are initiated from GFM. So not applicable for GFM itself
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d FM_EVENT_GLOBAL_FM_CONNECT", nodeId);
            break;
        }
        case FM_EVENT_GLOBAL_FM_DISCONNECT: {
            // configs are initiated from GFM. So not applicable for GFM itself
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d FM_EVENT_GLOBAL_FM_DISCONNECT", nodeId);
            break;
        }
    }
}

// TODO: need to check against restart,
//       and make sure the system is not re-initialized at restart.
FMIntReturn_t
FMFabricConfig::configOneNode( uint32_t nodeId, std::set<enum PortType> &portTypes )
{
    FMIntReturn_t rc;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d", nodeId);

    rc = configLwswitches( nodeId, portTypes );
    if ( rc != FM_INT_ST_OK )
    {
        return rc;
    }

    rc = configGpus( nodeId );
    return rc;
}

/*******************************************************************************************
 Note: All the required config information should be passed to this function as when this 
 function is called, the topology file is not yet opened/processed
*******************************************************************************************/
FMIntReturn_t
FMFabricConfig::sendNodeGlobalConfig( uint32_t nodeId )
{
    FMIntReturn_t rc;
    lwswitch::nodeGlobalConfigRequest *pGlobalConfigReq = NULL;
    lwswitch::fmMessage *pMessage = NULL;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_NODE_GLOBAL_CONFIG_REQ );

    pGlobalConfigReq = new lwswitch::nodeGlobalConfigRequest;
    pGlobalConfigReq->set_localnodeid( nodeId );
    pMessage->set_allocated_globalconfigrequest( pGlobalConfigReq );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send global config message to " NODE_ID_LOG_STR " %d failed with error %d\n", nodeId, rc);
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configOneLwswitch( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   std::set<enum PortType> &portTypes )
{
    FMIntReturn_t rc;

    rc = configSwitchPorts( nodeId, pSwitchInfo, portTypes );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    rc = configIngressReqTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    rc = configIngressRespTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    rc = configGangedLinkTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    // routing tables
    rc = configRmapTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    rc = configRidTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    rc = configRlanTable( nodeId, pSwitchInfo );
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
        return rc;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configSwitchPorts( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   std::set<enum PortType> &portTypes )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::switchPortConfigRequest *pConfigRequest = NULL;
    lwswitch::switchPortInfo *pCfg, *pMsg;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    int portIndex;
    PortKeyType key;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d",
                nodeId, pSwitchInfo->switchphysicalid());

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
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
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send switch port config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    return rc;
}

FMIntReturn_t
FMFabricConfig::configSwitchPortsWithTypes( std::set<enum PortType> &portTypes )
{
    FMIntReturn_t rc;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;
    SwitchKeyType key;

    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end();
          it++ )
    {
        key = it->first;
        rc = configSwitchPorts( key.nodeId, it->second, portTypes );
        if ( rc != FM_INT_ST_OK )
        {
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configSwitchPortList( uint32_t nodeId, uint32_t partitionId,
                                      std::list<PortKeyType> &portList, bool sync )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::switchPortConfigRequest *pConfigRequest = NULL;
    lwswitch::switchPortInfo *pCfg, *pMsg;
    std::list<PortKeyType>::iterator portListIt;

    for ( portListIt = portList.begin(); portListIt != portList.end(); portListIt++ )
    {
        PortKeyType key = *portListIt;

        map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
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

        if ( !pCfg || !pCfg->has_config())
        {
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
            pConfigRequest->set_switchphysicalid( key.physicalId );
        }

        pMsg = pConfigRequest->add_info();
        pMsg->CopyFrom( *pCfg );
        pMsg->set_partitionid( partitionId );
    }

    if ( pMessage )
    {
        if (sync)
        {
            lwswitch::fmMessage *pResponse = NULL;
            rc = SendMessageToLfmSync( nodeId, partitionId, pMessage,
                                       &pResponse, mCfgMsgTimeoutSec );
            if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
            {
                // failed to send the sync message
                FM_LOG_ERROR("request to send port config message to " NODE_ID_LOG_STR " %d failed with error %d",
                             nodeId, rc);
            }
            else if ( handleSWPortConfigRespMsg( pResponse, false ) == false )
            {
                // the response message carried error response
                FM_LOG_ERROR("failed to config port on " NODE_ID_LOG_STR " %d", nodeId);
                rc = FM_INT_ST_ILWALID_PORT_CFG;
            }
            if ( pResponse ) delete pResponse;
        }
        else
        {
            rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
            if ( rc != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("request to send switch port config to " NODE_ID_LOG_STR " %d partition %d with error %d",
                            nodeId, partitionId, rc);
            }
        }
    }
    return rc;
}

FMIntReturn_t
FMFabricConfig::configLwswitches( uint32_t nodeId, std::set<enum PortType> &portTypes )
{
    FMIntReturn_t rc;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d", nodeId);

    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end(); it++ )
    {
        SwitchKeyType key = it->first;
        lwswitch::switchInfo* pSwitchInfo = it->second;
        lwswitch::SwitchDegradedReason reasonNotUsed;
        if (mpGfm->mpDegradedModeMgr->isSwitchDegraded(key.nodeId, key.physicalId, reasonNotUsed)) {
            FM_LOG_INFO("not configuring LWSwitch " NODE_ID_LOG_STR " %d physical id %d as it is marked as degraded", key.nodeId, key.physicalId);
            continue;
        }

        if (nodeId != key.nodeId) {
            continue;
        }

        rc = configOneLwswitch( nodeId, pSwitchInfo, portTypes );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to configure LWSwitch for " NODE_ID_LOG_STR " %d LWSwitch physical id %d with error %d",
                        nodeId, key.physicalId, rc);
            return rc;
        }

        FM_LOG_INFO("LWSwitch %d/%d is configured", nodeId, key.physicalId);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configGpus( uint32_t nodeId )
{
    FMIntReturn_t rc;
    std::map <GpuKeyType, lwswitch::gpuInfo * >::iterator it;
    std::list<uint32_t> gpuPhysicalIdList;
    std::list<uint32_t>::iterator git;
    PartitionInfo partInfo;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d", nodeId);

    for ( it = mpGfm->mpParser->gpuCfg.begin();
          it != mpGfm->mpParser->gpuCfg.end(); it++ )
    {
        GpuKeyType key = it->first;
        if ( key.nodeId == nodeId )
        {
            FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, gpuPhysicalId %d",
                        nodeId, key.physicalId );
            gpuPhysicalIdList.push_back( key.physicalId );
        }
    }

    rc = configGpus(nodeId, ILWALID_FABRIC_PARTITION_ID, partInfo, gpuPhysicalIdList, true);
    if ( rc != FM_INT_ST_OK ) {
            FM_LOG_ERROR("failed to configure GPUs for " NODE_ID_LOG_STR " %d with error %d", nodeId, rc);
            return rc;
    }

    for ( git = gpuPhysicalIdList.begin(); git != gpuPhysicalIdList.end(); git++ )
    {
        FM_LOG_INFO("GPU %d/%d is configured", nodeId, *git);
    }

    return FM_INT_ST_OK;
}

// Generate FM message to only config GPUs
// on a specified shared lwswitch partition

lwswitch::fmMessage *
FMFabricConfig::generateConfigGpusMsg(uint32_t nodeId, uint32_t partitionId,
                                      PartitionInfo &partInfo,
                                      std::list<uint32_t> &gpuPhysicalIdList,
                                      lwswitch::FabricManagerMessageType configGPUMsgType,
                                      bool activate)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::gpuConfigRequest *pConfigRequest = NULL;
    lwswitch::gpuDetachRequest *pDeactivateRequest = NULL;
    lwswitch::gpuInfo *pCfg, *pMsg;
    std::map <GpuKeyType, lwswitch::gpuInfo * >::iterator it;
    GpuKeyType gpuKey;
    uint32_t gpuCount = 0;

    if (configGPUMsgType != lwswitch::FM_GPU_CONFIG_REQ)
    {
        FM_LOG_ERROR("invalid GPU config message type %d while generating GPU config message", configGPUMsgType);
        return pMessage;
    }

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type(configGPUMsgType);
    gpuKey.nodeId = nodeId;

    std::list<uint32_t>::iterator git;
    for ( git = gpuPhysicalIdList.begin(); git != gpuPhysicalIdList.end(); git++ )
    {
        gpuKey.physicalId = *git;
        pMsg = NULL;

        FM_LOG_DEBUG("generateConfigGpusMsg: GPU " NODE_ID_LOG_STR " %d physical id %d", nodeId, gpuKey.physicalId);

        lwswitch::GpuDegradedReason reasonNotUsed;
        if (mpGfm->mpDegradedModeMgr->isGpuDegraded(gpuKey.nodeId, gpuKey.physicalId, reasonNotUsed)) {
            FM_LOG_INFO("not configuring gpu " NODE_ID_LOG_STR " %d physical id %d as it is marked as degraded", gpuKey.nodeId, gpuKey.physicalId);
            continue;
        }

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
            FM_LOG_ERROR("invalid gpu configuration in topology file for " NODE_ID_LOG_STR " %d gpu id %d", nodeId, gpuKey.physicalId);
            continue;
        }

        if ( !pConfigRequest )
        {
            pConfigRequest = new lwswitch::gpuConfigRequest;
            pMessage->set_allocated_gpuconfigreq( pConfigRequest );
            pConfigRequest->set_partitionid( partitionId );
            pConfigRequest->set_config ( activate );
        }

        if ( partitionId == ILWALID_FABRIC_PARTITION_ID ) {
            FM_LOG_DEBUG("generateConfigGpusMsg: ILWALID_FABRIC_PARTITION_ID");

            // in bare metal
            // gpu uuid is obtained from topology discovery info
            char uuid[FM_UUID_BUFFER_SIZE];
            memset(uuid, 0, FM_UUID_BUFFER_SIZE);
            if (mpGfm->getGpuUuid(nodeId, gpuKey.physicalId, uuid))
            {
                pMsg = pConfigRequest->add_info();
                pMsg->CopyFrom( *pCfg );
                pMsg->set_uuid( uuid );

                pMsg->set_flaaddressrange(
                        FMDeviceProperty::getAddressRangePerGpu(mpGfm->getSwitchArchType()) );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
                FMGpuInfo_t devGpuInfo;
                if ( mpGfm->getGpuInfo( uuid, devGpuInfo ) )
                {
                    if ( devGpuInfo.isEgmCapable )
                    {
                        // the GPU is EGM capable
                        // GPA EGM and FLA EGM should be in the config already

                        // set FLA range with EGM
                        pMsg->set_flaaddressrange(
                                FMDeviceProperty::getEgmAddressRangePerGpu(mpGfm->getSwitchArchType()));
                        }
                    else
                    {
                        // the GPU is not EGM capable
                        // clear GPA EGM from the config
                        if ( pMsg->has_gpaegmaddressbase() )
                        {
                            pMsg->clear_gpaegmaddressbase();
                        }

                        if ( pMsg->has_gpaegmaddressrange() )
                        {
                            pMsg->clear_gpaegmaddressrange();
                        }

                        // set FLA range without EGM
                        pMsg->set_flaaddressrange(
                                FMDeviceProperty::getAddressRangePerGpu(mpGfm->getSwitchArchType()));
                    }
                }
#endif
            }
            else
            {
                // there is no LWLink mapping for this GPU
                // possible enabledLinkMask is 0 such in the case of MIG
                // no need to configure the GPU
                continue;
            }
        } else {
            switch (mpGfm->getFabricMode()) {
                case FM_MODE_SHARED_LWSWITCH:
                {
                    FM_LOG_DEBUG("generateConfigGpusMsg: Shared LWSwitch");
                    //
                    // For Shared LWSwitch mode
                    // topology discovery info is not available anymore
                    // gpu uuid is obtained from partition info
                    //
                    PartitionGpuInfoList::iterator jit;
                    for ( jit = partInfo.gpuInfo.begin(); jit != partInfo.gpuInfo.end(); jit++ )
                    {
                        PartitionGpuInfo gpuInfo = *jit;
                        if ( gpuInfo.physicalId == gpuKey.physicalId )
                        {
                            pMsg = pConfigRequest->add_info();
                            pMsg->CopyFrom( *pCfg );
                            pMsg->set_uuid( gpuInfo.uuid );

                            pMsg->set_flaaddressrange(
                            FMDeviceProperty::getAddressRangePerGpu(mpGfm->getSwitchArchType()) );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
                            FMGpuInfo_t devGpuInfo;
                            //
                            // GlobalFabricManager::getGpuInfo(char uuid[], FMGpuInfo_t &gpuInfo) does not
                            // rely on topology discovery, it relies on GPU uuid only.
                            //
                            if ( mpGfm->getGpuInfo( gpuInfo.uuid, devGpuInfo) )
                            {
                                if ( devGpuInfo.isEgmCapable )
                                {
                                    // the GPU is EGM capable
                                    // GPA EGM and FLA EGM should be in the config already

                                    // set FLA range with EGM
                                    pMsg->set_flaaddressrange(
                                            FMDeviceProperty::getEgmAddressRangePerGpu(mpGfm->getSwitchArchType()));
                                    }
                                else
                                {
                                    // the GPU is not EGM capable
                                    // clear GPA EGM from the config
                                    if ( pMsg->has_gpaegmaddressbase() )
                                    {
                                        pMsg->clear_gpaegmaddressbase();
                                    }

                                    if ( pMsg->has_gpaegmaddressrange() )
                                    {
                                        pMsg->clear_gpaegmaddressrange();
                                    }

                                    // set FLA range without EGM
                                    pMsg->set_flaaddressrange(
                                            FMDeviceProperty::getAddressRangePerGpu(mpGfm->getSwitchArchType()));
                                }
                            }
#endif
                            break;
                        }
                    }
                    break;
                }
                case FM_MODE_VGPU:
                {
                    FM_LOG_DEBUG("generateConfigGpusMsg: vGPU");

                    //
                    // For vGPU mode
                    //
                    // 1. Skip programming of GPA address and its range.
                    // 2. Program overlap FLA address and its range i.e FLA base for the
                    //    first GPU in each partition will start from the first GPU's
                    //    (node 0) FLA base in the topology file.
                    //
                    PartitionGpuInfoList::iterator jit;
                    int i = 0; // First GPU's physical ID in the topology file
                    
                    for (jit = partInfo.gpuInfo.begin(), i = 0; jit != partInfo.gpuInfo.end(); jit++, i++)
                    {
                        PartitionGpuInfo gpuInfo = *jit;

                        if (gpuInfo.physicalId == gpuKey.physicalId)
                        {
                            uint64_t flaAddress = 0;
                            lwswitch::gpuInfo *gCfg;
                            pMsg = pConfigRequest->add_info();
                            pMsg->CopyFrom( *pCfg );
                            pMsg->set_uuid( gpuInfo.uuid );

                            // Clear GPA address and its range
                            pMsg->set_gpaaddressbase(0);
                            pMsg->set_gpaaddressrange(0);

                            // Program overlap FLA address
                            flaAddress = FMDeviceProperty::getFlaFromTargetId(mpGfm->getSwitchArchType(), GPU_TARGET_ID(0, i));
                            pMsg->set_flaaddressbase(flaAddress);

                            FM_LOG_DEBUG("generateConfigGpusMsg: physicalid %d FLA 0x%lx RANGE 0x%lx",
                                         gpuInfo.physicalId, pMsg->flaaddressbase(), pMsg->flaaddressrange());
                            break;
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
        gpuCount++;
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
FMIntReturn_t
FMFabricConfig::configGpus(uint32_t nodeId, uint32_t partitionId,
                           PartitionInfo &partInfo,
                           std::list<uint32_t> gpuPhysicalIdList,
                           bool activate)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = generateConfigGpusMsg(nodeId, partitionId,
                                                          partInfo,
                                                          gpuPhysicalIdList,
                                                          lwswitch::FM_GPU_CONFIG_REQ, activate);
    if ( pMessage )
    {
        // get a local copy of the message for error handling as the original message will be freed in sendMessageToLfm.
        lwswitch::fmMessage errMsg = *pMessage;
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send GPU config request message to " NODE_ID_LOG_STR " %d fabric partition id %d failed with error %d",
                        nodeId, partitionId, rc);

            if ( partitionId == ILWALID_FABRIC_PARTITION_ID ) {
                handleConfigError( nodeId, ERROR_SOURCE_SW_GLOBALFM,
                                   ERROR_TYPE_CONFIG_GPU_FAILED, errMsg);
            }
        }
    }

    return rc;
}

// Config a single GPU
FMIntReturn_t
FMFabricConfig::configGpu( uint32_t nodeId, char *gpuUuid)
{
    uint32_t physicalId;
    if ( !gpuUuid ||
         ( mpGfm->getGpuPhysicalId(nodeId, gpuUuid, physicalId) == false ) )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " %d GPU %s.", nodeId, gpuUuid ? gpuUuid : "Null UUID");
        return FM_INT_ST_ILWALID_GPU;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
    PartitionInfo partInfo;
    std::list<uint32_t> gpuPhysicalIdList;
    gpuPhysicalIdList.clear();
    gpuPhysicalIdList.push_back( physicalId );
    lwswitch::fmMessage *pMessage = generateConfigGpusMsg(nodeId,
                                                          ILWALID_FABRIC_PARTITION_ID,
                                                          partInfo,
                                                          gpuPhysicalIdList,
                                                          lwswitch::FM_GPU_CONFIG_REQ, true);
    if ( !pMessage )
    {
        // failed to construct the gpu config fm message
        // error already logged
        return FM_INT_ST_GENERIC_ERROR;
    }

    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                               &pResponse, mCfgMsgTimeoutSec );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU config message to " NODE_ID_LOG_STR " %d failed with error %d",
                     nodeId, rc);
    }
    else if ( handleGpuConfigRespMsg( pResponse, false ) == false )
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to config GPU " NODE_ID_LOG_STR " %d UUID %s", nodeId, gpuUuid);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

// attach GPUs on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
FMIntReturn_t
FMFabricConfig::attachGpus( uint32_t nodeId, PartitionInfo &partInfo )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuAttachRequest *pAttachRequest = new lwswitch::gpuAttachRequest();

    pAttachRequest->set_partitionid( partInfo.partitionId );

    /* 
        Fill in information, namely uuid of each gpu that needs to be attached
    */

    PartitionGpuInfoList::iterator it;
    for ( it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++ )
    {
        PartitionGpuInfo gpuInfo = *it;
        lwswitch::gpuAttachRequestInfo *pAttachReqInfo = pAttachRequest->add_info();
        pAttachReqInfo->set_uuid( gpuInfo.uuid );
        pAttachReqInfo->set_gpuphysicalid(gpuInfo.physicalId);
    }

    // send the final messages to LFM
    pMessage->set_type( lwswitch::FM_GPU_ATTACH_REQ );
    pMessage->set_allocated_gpuattachreq( pAttachRequest );

    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync( nodeId, partInfo.partitionId, pMessage,
                               &pResponse, (mCfgMsgTimeoutSec*2) );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU attach message to " NODE_ID_LOG_STR " %d failed with error %d",
                     nodeId, rc);
    }
    else if ( handleGpuAttachRespMsg( pResponse, false ) == false )
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to attach GPU " NODE_ID_LOG_STR " %d partition id %d", nodeId, partInfo.partitionId);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

// detach GPUs on a specified shared lwswitch partition
// partitionId is only used for activate or deactivate GPUs.
// partInfo will be NULL in cases where we need to detach all GPUs
FMIntReturn_t
FMFabricConfig::detachGpus( uint32_t nodeId, PartitionInfo *partInfo)
{
    FMIntReturn_t rc = FM_INT_ST_OK;

    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();;
    lwswitch::gpuDetachRequest *pDetachRequest = new lwswitch::gpuDetachRequest();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    /* 
        Fill in information, namely uuid of each gpu that needs to be detached
    */
    /*
        The partInfo != NULL check is done to differentiate between per GPU detach
        and detaching all GPUs 
    */
    if (partInfo != NULL) {
        partitionId = partInfo->partitionId;
        pDetachRequest->set_partitionid( partitionId );
        PartitionInfo pInfo = *partInfo;

        PartitionGpuInfoList::iterator it;
        for ( it = pInfo.gpuInfo.begin(); it != pInfo.gpuInfo.end(); it++ )
        {
            PartitionGpuInfo gpuInfo = *it;
            // fill the disable link GPB message.
            lwswitch::gpuDetachRequestInfo *pDeactivateRequestInfo = pDetachRequest->add_info();
            pDeactivateRequestInfo->set_uuid( gpuInfo.uuid );
            pDeactivateRequestInfo->set_gpuphysicalid(gpuInfo.physicalId);
        }
    }

    // send the final messages to LFM
    pMessage->set_type( lwswitch::FM_GPU_DETACH_REQ );
    pMessage->set_allocated_gpudetachreq( pDetachRequest );

    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync( nodeId, partitionId, pMessage,
                               &pResponse, (mCfgMsgTimeoutSec*2) );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU detach message to " NODE_ID_LOG_STR " %d failed with error %d",
                     nodeId, rc);
    }
    else if ( handleGpuDetachRespMsg( pResponse, false ) == false )
    {
        // the response message carried error response
        if ( partitionId != ILWALID_FABRIC_PARTITION_ID )
        {
            FM_LOG_ERROR("failed to detach GPU " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
        }
        else
        {
            FM_LOG_ERROR("failed to detach GPUs on " NODE_ID_LOG_STR " %d", nodeId);
        }
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

FMIntReturn_t
FMFabricConfig::configIngressReqEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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

    for ( key.index = 0; key.index < FMDeviceProperty::getIngressReqTableSize(mpGfm->getSwitchArchType()); key.index++ )
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
            FM_LOG_ERROR("invalid ingress request entries for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d index %d",
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

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        FM_LOG_DEBUG("portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send ingress request config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FMIntReturn_t
FMFabricConfig::configIngressRespEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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

    for ( key.index = 0; key.index < FMDeviceProperty::getIngressRespTableSize(mpGfm->getSwitchArchType()); key.index++ )
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
            FM_LOG_ERROR("invalid ingress response entries for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d index %d",
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

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        FM_LOG_DEBUG("portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send ingress response config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                        nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);
    return rc;
}

FMIntReturn_t
FMFabricConfig::configGangedLinkEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo, int portIndex )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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
        return FM_INT_ST_OK;
    }

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();
    key.portIndex  = portIndex;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_GANGED_LINK_TABLE_REQ );

    pConfigRequest = new lwswitch::switchPortGangedLinkTable;
    pMessage->set_allocated_gangedlinktablerequest( pConfigRequest );
    pConfigRequest->set_switchphysicalid( pSwitchInfo->switchphysicalid() );

    for ( key.index = 0; key.index < FMDeviceProperty::getGangedLinkTableSize(mpGfm->getSwitchArchType()); key.index++ )
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
            FM_LOG_ERROR("invalid ganged link entries for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d index %d",
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
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send ganged link config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                          nodeId, pSwitchInfo->switchphysicalid(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);
    return rc;
}

FMIntReturn_t
FMFabricConfig::configIngressReqTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    if ( mpGfm->mpParser->reqEntry.size() == 0 )
    {
        // there is no ingress request entry to configure
        return FM_INT_ST_OK;
    }

    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configIngressReqEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::sendPeerLFMInfo( uint32_t nodeId )
{
    // send all other Node's ID and IP information to the peer
    // for establishing peer LFM connections
    if (mpGfm->mpParser->NodeCfg.size() < 2)
    {
        // nothing to do if we have only one node, ie no LFM to LFM connection needed
        return FM_INT_ST_OK;
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

    FMIntReturn_t rc;
    lwswitch::fmMessage *pMessage = NULL;
    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_NODE_INFO_MSG );
    pMessage->set_allocated_nodeinfomsg( nodeInfoMsg );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send peer fabric manager information to " NODE_ID_LOG_STR " %d failed with error %d", nodeId, rc);
    }

    return rc;
}

// send LFM FM_CONFIG_INIT_DONE_REQ synchronously
FMIntReturn_t
FMFabricConfig::sendConfigInitDoneReqMsg( uint32_t nodeId )
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::configInitDoneReq *pInitDoneReq;

    // prepare the response message    
    pInitDoneReq = new lwswitch::configInitDoneReq();

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_CONFIG_INIT_DONE_REQ );
    pMessage->set_allocated_initdonereq( pInitDoneReq );

    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                               &pResponse, mCfgMsgTimeoutSec );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send config init done message to " NODE_ID_LOG_STR " %d failed with error %d",
                     nodeId, rc);
    }
    else if ( handleConfigInitDoneRespMsg( pResponse ) == false )
    {
        // error already logged
        rc = FM_INT_ST_ILWALID_PORT_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

FMIntReturn_t
FMFabricConfig::sendConfigDeInitReqMsg( uint32_t nodeId )
{
    FMIntReturn_t rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::configDeInitReq *pDeInitReq;

    // prepare the response message    
    pDeInitReq = new lwswitch::configDeInitReq();

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_CONFIG_DEINIT_REQ );
    pMessage->set_allocated_deinitreq( pDeInitReq );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send config deinit done message to " NODE_ID_LOG_STR " %d failed with error %d", nodeId, rc);
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configIngressRespTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    if ( mpGfm->mpParser->respEntry.size() == 0 )
    {
        // there is no ingress response entry to configure
        return FM_INT_ST_OK;
    }

    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configIngressRespEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configGangedLinkTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    if ( mpGfm->mpParser->gangedLinkEntry.size() == 0 )
    {
        // there is no ganged link table entry to configure
        return FM_INT_ST_OK;
    }

    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configGangedLinkEntries( nodeId, pSwitchInfo, key.portIndex );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

// activate or deactivate ingress request tables entries for a
// specified shared lwswitch partition
// ingress request entries are set to valid when the partition is activated
// ingress request entries are set to invalid when the partition is deactivated
FMIntReturn_t
FMFabricConfig::configIngressReqTable( uint32_t nodeId, uint32_t partitionId,
                                       uint32_t switchPhysicalId, uint32_t portNum,
                                       std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    if ( mpGfm->mpParser->reqEntry.size() == 0 )
    {
        // there is no ingress response entry to configure
        return FM_INT_ST_OK;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
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
              key.index < ingressReqIndex + FMDeviceProperty::getNumIngressReqEntriesPerGpu(mpGfm->getSwitchArchType());
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

                FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                             portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
            }
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send ingress request config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d for partition %s failed with error %d",
                          nodeId, switchPhysicalId, portNum, strOperation.c_str(), rc);
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
FMIntReturn_t
FMFabricConfig::configIngressRespTable( uint32_t nodeId, uint32_t partitionId,
                                        uint32_t switchPhysicalId, uint32_t portNum,
                                        std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    if ( mpGfm->mpParser->respEntry.size() == 0 )
    {
        // there is no ingress response entry to configure
        return FM_INT_ST_OK;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
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
        uint32_t ingressRespIndex = gpuEndpointID * FMDeviceProperty::getNumIngressRespEntriesPerGpu(mpGfm->getSwitchArchType());

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_firstindex( ingressRespIndex );
        info->set_partitionid( partitionId );

        for ( key.index = ingressRespIndex;
              key.index < ingressRespIndex + FMDeviceProperty::getNumIngressRespEntriesPerGpu(mpGfm->getSwitchArchType());
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

                FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                             portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
            }
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send ingress response config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d for partition %s failed with error %d",
                          nodeId, switchPhysicalId, portNum, strOperation.c_str(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRmapTableByAddrType( uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                           uint32_t portNum, std::list<uint32_t> &gpuPhysicalIds, bool activate,
                                           FabricAddrType addrType)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRmapTableRequest *pConfigRequest;
    lwswitch::portRmapTableInfo *info = NULL;
    rmapPolicyEntry *pCfg, *pMsg;
    RemapTable remapTable;
    std::map <RmapTableKeyType, rmapPolicyEntry * >::iterator it;
    RmapTableKeyType key;
    int count = 0;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RMAP_TABLE_REQ );

    pConfigRequest = new lwswitch::portRmapTableRequest;
    pMessage->set_allocated_rmaptablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( switchPhysicalId );

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for ( std::list<uint32_t>::iterator it = gpuPhysicalIds.begin();
          it != gpuPhysicalIds.end(); it++ )
    {
        uint32_t gpuPhysicalId = *it;
        uint32_t index;
        uint32_t targetId = nodeId * MAX_NUM_GPUS_PER_NODE + gpuPhysicalId;

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_partitionid( partitionId );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        FMGpuInfo_t gpuInfo;

        uint32_t egmGpaIndex = 0, egmFlaIndex = 0;
        uint32_t spaIndex = 0;
        bool isEgmCapable = false;
        bool isSpaCapable = false;

        if ( mpGfm->getGpuInfo( nodeId, gpuPhysicalId, gpuInfo) )
        {
            isEgmCapable = gpuInfo.isEgmCapable;
            egmGpaIndex = FMDeviceProperty::getFlaEgmRemapIndexFromTargetId(mpGfm->getSwitchArchType(), targetId);
            egmFlaIndex = FMDeviceProperty::getFlaEgmRemapIndexFromTargetId(mpGfm->getSwitchArchType(), targetId);

            isSpaCapable = gpuInfo.isSpaCapable;
            spaIndex = FMDeviceProperty::getSpaRemapIndexFromSpaAddress(mpGfm->getSwitchArchType(), gpuInfo.spaAddress);
        }
#endif
        switch (addrType)
        {
            case GPA_ADDR_TYPE:
            {
                // GPA entries
                remapTable = FMDeviceProperty::FMDeviceProperty::getGpaRemapTbl(mpGfm->getSwitchArchType());
                index = FMDeviceProperty::getGpaRemapIndexFromTargetId(mpGfm->getSwitchArchType(), targetId);
                info->set_firstindex( index );
                info->set_table(remapTable);

                for ( key.index = index;
                      key.index < index + FMDeviceProperty::getNumGpaRemapEntriesPerGpu(mpGfm->getSwitchArchType());
                      key.index++ )
                {
                    pCfg = getRmapEntryByTable( key, remapTable );
                    if ( pCfg )
                    {
                        pMsg = info->add_entry();
                        pMsg->CopyFrom( *pCfg );

                        // modify the valid bit
                        pMsg->set_entryvalid( activate ? 1 : 0 );
                        count++;

                        FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                                     portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
                    }
                }
                break;
            }
            case FLA_ADDR_TYPE:
            {
                // FLA entries
                remapTable = FMDeviceProperty::getFlaRemapTbl(mpGfm->getSwitchArchType());
                index = FMDeviceProperty::getFlaRemapIndexFromTargetId(mpGfm->getSwitchArchType(), targetId);
                info->set_firstindex( index );
                info->set_table(remapTable);

                for ( key.index = index;
                      key.index < index + FMDeviceProperty::getNumFlaRemapEntriesPerGpu(mpGfm->getSwitchArchType());
                      key.index++ )
                {
                    pCfg = getRmapEntryByTable( key, remapTable );
                    if ( pCfg )
                    {
                        pMsg = info->add_entry();
                        pMsg->CopyFrom( *pCfg );

                        // modify the valid bit
                        pMsg->set_entryvalid( activate ? 1 : 0 );
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
                        // set EGM remap entries invalid if the GPU is not EGM capable
                        if ( !isEgmCapable &&
                             ( (key.index == egmGpaIndex) || (key.index == egmFlaIndex) ) ) {
                            pMsg->set_entryvalid( 0 );
                        }
#endif
                        count++;

                        FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                                     portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
                    }
                }
                break;
            }
    #if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            case SPA_ADDR_TYPE:
            {
                // program SPA remap entries only if the GPU is SPA capable
                remapTable = FMDeviceProperty::getSpaRemapTbl(mpGfm->getSwitchArchType());
                index = spaIndex;
                info->set_firstindex( index );
                info->set_table( remapTable );

                for ( key.index = index;
                      key.index < index + FMDeviceProperty::getNumSpaRemapEntriesPerGpu(mpGfm->getSwitchArchType());
                      key.index++ )
                {
                    pCfg = getRmapEntryByTable( key, remapTable );
                    if ( pCfg )
                    {
                        pMsg = info->add_entry();
                        pMsg->CopyFrom( *pCfg );

                        // set the targetId
                        pMsg->set_targetid(targetId);

                        // modify the valid bit
                        pMsg->set_entryvalid( activate ? 1 : 0 );

                        // set SPA remap entries invalid if the GPU is not SPA capable
                        if ( !isSpaCapable )
                        {
                            pMsg->set_entryvalid( 0 );
                        }

                        count++;

                        FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                                     portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
                    }
                }

                break;
            }
#endif
            default:
                break;
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send remap request config to " NODE_ID_LOG_STR " %d for partition %s failed with error %d",
                         nodeId, strOperation.c_str(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRmapTableWithVfs(uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                       uint32_t portNum, PartitionInfo &partInfo, bool activate)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRmapTableRequest *pConfigRequest;
    lwswitch::portRmapTableInfo *info = NULL;
    PartitionGpuInfoList::iterator it;
    rmapPolicyEntry *pCfg, *pMsg;
    RmapTableKeyType key;
    int count = 0, i = 0;
    accessPort *accessPortInfo = mpGfm->mpParser->getAccessPortInfo(nodeId, switchPhysicalId, portNum);
    uint32_t srcGfid = 0;

    FM_LOG_DEBUG("configRmapTableWithVfs: " NODE_ID_LOG_STR " %d partion id %d switch ID %d port %d activate %d",
                 nodeId, partitionId, switchPhysicalId, portNum, activate);

    // Find Source GPU to compute its GFID
    for (it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++)
    {
        PartitionGpuInfo gpuinfo = *it;

        // Compute source gpu's GFID 
        if (accessPortInfo && accessPortInfo->has_farpeerid() &&
            (accessPortInfo->farpeerid() == gpuinfo.physicalId))
        {
            srcGfid = gpuinfo.dynamicInfo.gfidInfo.gfid;
            break;
        }
    }

    // Return failure if it is a Rmap table config request and source GPU GFID is Zero
    if (activate && (srcGfid == 0)) {
        FM_LOG_ERROR("failed to configure rmap table for partition id %d since source gfid is zero", partitionId);
        return FM_INT_ST_NOT_CONFIGURED;
    }

    RemapTable remapTable = FMDeviceProperty::getFlaRemapTbl(mpGfm->getSwitchArchType());
    pMessage = new lwswitch::fmMessage();
    pMessage->set_type(lwswitch::FM_RMAP_TABLE_REQ);

    pConfigRequest = new lwswitch::portRmapTableRequest;
    pMessage->set_allocated_rmaptablereq(pConfigRequest);
    pConfigRequest->set_switchphysicalid(switchPhysicalId);

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for (it = partInfo.gpuInfo.begin(), i = 0; it != partInfo.gpuInfo.end(); it++, i++)
    {
        PartitionGpuInfo gpuinfo = *it;
        uint32_t targetId = GPU_TARGET_ID(nodeId, gpuinfo.physicalId);
        uint32_t dstGfid = gpuinfo.dynamicInfo.gfidInfo.gfid;

        // Skip programming of Rmap table if it is a source GPU
        if (accessPortInfo->farpeerid() == gpuinfo.physicalId)
            continue;

        //
        // vGPU Mode - FLA Address Programming
        // FLA base for the first GPU in each partition will start from the first GPU's (node 0) FLA base in the topology file.
        //
        uint32_t index = FMDeviceProperty::getFlaRemapIndexFromTargetId(mpGfm->getSwitchArchType(), GPU_TARGET_ID(0, i));
        info = pConfigRequest->add_info();
        info->set_switchphysicalid(switchPhysicalId);
        info->set_port(portNum);
        info->set_firstindex(index);
        info->set_partitionid(partitionId);

        for (key.index = index;
             key.index < index + FMDeviceProperty::getNumFlaRemapEntriesPerGpu(mpGfm->getSwitchArchType());
             key.index++)
        {
            std::map <RmapTableKeyType, rmapPolicyEntry *>::iterator entryIt;
            entryIt = mpGfm->mpParser->rmapEntry.find(key);
            if (entryIt != mpGfm->mpParser->rmapEntry.end())
            {
                pCfg = entryIt->second;
                pMsg = info->add_entry();
                pMsg->CopyFrom(*pCfg);

                FM_LOG_DEBUG("configRmapTableWithVfs: activate %d physicalid %x targetId %x srcGfid %x dstGfid %x FLA ADDRESS 0x%lx",
                             activate, gpuinfo.physicalId, targetId, srcGfid, dstGfid, pMsg->address());

                // Update Remap tables based on activation/deactivation request
                if (activate) {
                    pMsg->set_entryvalid(1);
                    pMsg->set_targetid (targetId);           
                    pMsg->set_reqcontextchk(srcGfid);
                    pMsg->set_reqcontextmask(gpuinfo.dynamicInfo.gfidInfo.gfidMask);
                    // Preserve FLA address bit, as reqcontextrep replaces the entire 16 bits.
                    pMsg->set_reqcontextrep(REMAP_REQCTX_FLA_BIT | dstGfid);
                    pMsg->set_remapflags(REMAP_FLAGS_REMAP_ADDR | REMAP_FLAGS_REQCTXT_CHECK | REMAP_FLAGS_REQCTXT_REPLACE);
                } else {
                    pMsg->set_entryvalid(0);
                }

                count++;
            }
        }
    }

    if (count > 0)
    {
        rc = SendMessageToLfm(nodeId, partitionId, pMessage, true);
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send remap request config to " NODE_ID_LOG_STR " %d for partition %s failed with error %d",
                nodeId, strOperation.c_str(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRmapTable( uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                 uint32_t portNum, std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    FMIntReturn_t rc;

    rc = configRmapTableByAddrType( nodeId, partitionId, switchPhysicalId, portNum, gpuPhysicalIds, activate, GPA_ADDR_TYPE );
    if (rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }

    rc = configRmapTableByAddrType( nodeId, partitionId, switchPhysicalId, portNum, gpuPhysicalIds, activate, FLA_ADDR_TYPE );
    if (rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    rc = configRmapTableByAddrType( nodeId, partitionId, switchPhysicalId, portNum, gpuPhysicalIds, activate, SPA_ADDR_TYPE );
    if (rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }
#endif
    return rc;
}

FMIntReturn_t
FMFabricConfig::configRidTable( uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                uint32_t portNum, std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRidTableRequest *pConfigRequest;
    lwswitch::portRidTableInfo *info = NULL;
    ridRouteEntry *pCfg, *pMsg;
    std::map <RidTableKeyType, ridRouteEntry * >::iterator it;
    RidTableKeyType key;
    int count = 0;
    accessPort *accessPortInfo = mpGfm->mpParser->getAccessPortInfo( nodeId, switchPhysicalId, portNum );

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RID_TABLE_REQ );

    pConfigRequest = new lwswitch::portRidTableRequest;
    pMessage->set_allocated_ridtablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( switchPhysicalId );

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for ( std::list<uint32_t>::iterator it = gpuPhysicalIds.begin();
          it != gpuPhysicalIds.end(); it++ )
    {
        uint32_t gpuPhysicalId = *it;

        uint32_t targetId = nodeId * MAX_NUM_GPUS_PER_NODE + gpuPhysicalId;
        uint32_t index = targetId;
        key.index = index;

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_firstindex( index );
        info->set_partitionid( partitionId );

        std::map <RidTableKeyType, ridRouteEntry *>::iterator entryIt;
        entryIt = mpGfm->mpParser->ridEntry.find(key);
        if ( entryIt != mpGfm->mpParser->ridEntry.end() )
        {
            pCfg = entryIt->second;
            pMsg = info->add_entry();
            pMsg->CopyFrom( *pCfg );
            removeDisabledPortsFromRidEntry( nodeId, switchPhysicalId, pMsg );

            // modify the valid bit
            pMsg->set_valid( activate ? 1 : 0 );
            count++;

            FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                         portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send rid request config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d for partition %s failed with error %d",
                          nodeId, switchPhysicalId, portNum, strOperation.c_str(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRlanTable( uint32_t nodeId, uint32_t partitionId, uint32_t switchPhysicalId,
                                 uint32_t portNum, std::list<uint32_t> &gpuPhysicalIds, bool activate )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage;
    lwswitch::portRlanTableRequest *pConfigRequest;
    lwswitch::portRlanTableInfo *info = NULL;
    rlanRouteEntry *pCfg, *pMsg;
    std::map <RlanTableKeyType, rlanRouteEntry * >::iterator it;
    RlanTableKeyType key;
    int count = 0;

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_RLAN_TABLE_REQ );

    pConfigRequest = new lwswitch::portRlanTableRequest;
    pMessage->set_allocated_rlantablereq( pConfigRequest );
    pConfigRequest->set_switchphysicalid( switchPhysicalId );

    key.nodeId = nodeId;
    key.portIndex = portNum;
    key.physicalId = switchPhysicalId;

    for ( std::list<uint32_t>::iterator it = gpuPhysicalIds.begin();
          it != gpuPhysicalIds.end(); it++ )
    {
        uint32_t gpuPhysicalId = *it;

        uint32_t targetId = nodeId * MAX_NUM_GPUS_PER_NODE + gpuPhysicalId;
        uint32_t index = targetId;
        key.index = index;

        info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portNum );
        info->set_firstindex( index );
        info->set_partitionid( partitionId );

        std::map <RlanTableKeyType, rlanRouteEntry *>::iterator entryIt;
        entryIt = mpGfm->mpParser->rlanEntry.find(key);
        if ( entryIt != mpGfm->mpParser->rlanEntry.end() )
        {
            pCfg = entryIt->second;
            pMsg = info->add_entry();
            pMsg->CopyFrom( *pCfg );
            removeDisabledPortsFromRlanEntry( nodeId, switchPhysicalId, pMsg );

            // modify the valid bit
            pMsg->set_valid( activate ? 1 : 0 );
            count++;

            FM_LOG_DEBUG("portNum %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                         portNum, info->firstindex(), key.index, pCfg->index(), pMsg->index());
        }
    }

    if ( count > 0 )
    {
        rc = SendMessageToLfm( nodeId, partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            std::string strOperation = activate ? "activation" : "deactivation";
            FM_LOG_ERROR("request to send rlan request config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d for partition %s failed with error %d",
                          nodeId, switchPhysicalId, portNum, strOperation.c_str(), rc);
        }
    }
    else
    {
        delete pMessage;
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configSharedLWSwitchPartitionRoutingTable( uint32_t nodeId, PartitionInfo &partInfo, bool activate )
{
    FMIntReturn_t rc = FM_INT_ST_OK;

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
        FMPciInfo_t pciInfo = {0};
        mpGfm->getLWSwitchPciBdf(nodeId, switchPhysicalId, pciInfo);
        std::string strOperation = activate ? "activation" : "deactivation";

        // program routing entries on enabled ports
        for ( uint32_t portNum = 0; portNum < LWLINK_MAX_DEVICE_CONN; portNum++ ) {
            // skip if the link is not enabled
            if ( !(switchinfo.enabledLinkMask & ((uint64)1 << portNum)) )
                continue;

            rc = configIngressReqTable( nodeId, partitionId, switchPhysicalId,
                                        portNum, gpuPhysicalIds, activate );
            if ( rc != FM_INT_ST_OK )
            {
                std::ostringstream ss;
                ss << "failed to configure request routing information for fabric partition id " << partitionId
                   << " LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id " << pciInfo.busId << " physical id " << switchPhysicalId << " port " << portNum
                   << " during partition " << strOperation;
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
            }

            rc = configIngressRespTable( nodeId, partitionId, switchPhysicalId,
                                         portNum, gpuPhysicalIds, activate );
            if ( rc != FM_INT_ST_OK )
            {
                std::ostringstream ss;
                ss << "failed to configure response routing information for fabric partition id " << partitionId 
                   << " LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id " << pciInfo.busId << " physical id " << switchPhysicalId << " port " << portNum
                   << " during partition " << strOperation;
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
            }

            if (mpGfm->getFabricMode() == FM_MODE_VGPU) {
                rc = configRmapTableWithVfs(nodeId, partitionId, switchPhysicalId,
                                            portNum, partInfo, activate);
            } else {
                rc = configRmapTable(nodeId, partitionId, switchPhysicalId,
                                     portNum, gpuPhysicalIds, activate);
            }

            if ( rc != FM_INT_ST_OK )
            {
                std::ostringstream ss;
                ss << "failed to configure remap routing information for fabric partition id " << partitionId
                   << " LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id " << pciInfo.busId << " physical id " << switchPhysicalId << " port " << portNum
                   << " during partition " << strOperation;
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
            }

            rc = configRidTable( nodeId, partitionId, switchPhysicalId,
                                 portNum, gpuPhysicalIds, activate );
            if ( rc != FM_INT_ST_OK )
            {
                std::ostringstream ss;
                ss << "failed to configure rid routing information for fabric partition id " << partitionId
                   << " LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id " << pciInfo.busId << " physical id " << switchPhysicalId << " port " << portNum
                   << " during partition " << strOperation;
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
            }

            rc = configRlanTable( nodeId, partitionId, switchPhysicalId,
                                  portNum, gpuPhysicalIds, activate );
            if ( rc != FM_INT_ST_OK )
            {
                std::ostringstream ss;
                ss << "failed to configure rlan routing information for fabric partition id " << partitionId
                   << " LWSwitch " << NODE_ID_LOG_STR << " " << nodeId << " pci bus id " << pciInfo.busId << " physical id " << switchPhysicalId << " port " << portNum
                   << " during partition " << strOperation;
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
            }
        }
    }

    return rc;
}

// configure the GPU fabric address for GPUs in a specified partition
FMIntReturn_t
FMFabricConfig::configSharedLWSwitchPartitionGPUs(uint32_t nodeId, PartitionInfo &partInfo, bool activate)
{
    std::list <uint32> gpuPhysicalIdList;

    // go through each GPU and create a list of physical id
    for ( PartitionGpuInfoList::iterator it = partInfo.gpuInfo.begin();
          it != partInfo.gpuInfo.end(); it++ )
    {
        PartitionGpuInfo& gpuinfo = *it;

        // do not config fabric address on the GPU
        // if the GPU has no LWLink enabled on it
        if ( gpuinfo.numEnabledLinks == 0 )
            continue;

        gpuPhysicalIdList.push_back( gpuinfo.physicalId );
    }

    FMIntReturn_t rc = configGpus(nodeId, partInfo.partitionId, partInfo, gpuPhysicalIdList, activate) ;
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("failed to configure GPUs for fabric partition with " NODE_ID_LOG_STR " %d and partition id %d", nodeId, partInfo.partitionId);
        FM_SYSLOG_ERR("failed to configure GPUs for fabric partition with " NODE_ID_LOG_STR " %d and partition id %d", nodeId, partInfo.partitionId);
    }

    return rc;
}

// attach a single GPU, and register GPU event
FMIntReturn_t
FMFabricConfig::configAttachGpu( uint32_t nodeId, char *gpuUuid )
{
    if ( !gpuUuid )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " %d GPU %s.", nodeId, gpuUuid ? gpuUuid : "Null UUID");
        return FM_INT_ST_ILWALID_GPU;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuAttachRequest *pAttachRequest = new lwswitch::gpuAttachRequest();

    pAttachRequest->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

    //Fill in information, namely uuid of each gpu that needs to be attached
    lwswitch::gpuAttachRequestInfo *pAttachReqInfo = pAttachRequest->add_info();
    pAttachReqInfo->set_uuid( gpuUuid );
    pAttachReqInfo->set_registerevent( true );

    uint32_t physicalId;
    if ( mpGfm->getGpuPhysicalId(nodeId, gpuUuid, physicalId) == false )
    {
        // physicalId is no longer required as attach/detach is based on UUID
        FM_LOG_DEBUG("cannot find mapping for " NODE_ID_LOG_STR " %d GPU %s.", nodeId, gpuUuid);
    }
    else
    {
        pAttachReqInfo->set_gpuphysicalid(physicalId);
    }

    // send the final messages to LFM
    pMessage->set_type( lwswitch::FM_GPU_ATTACH_REQ );
    pMessage->set_allocated_gpuattachreq( pAttachRequest );

    lwswitch::fmMessage *pResponse = NULL;

    rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                               &pResponse, mCfgMsgTimeoutSec );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU attach message to " NODE_ID_LOG_STR " %d failed with error %d",
                     nodeId, rc);
    }
    else if ( handleGpuAttachRespMsg( pResponse, false ) == false )
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to attach GPU " NODE_ID_LOG_STR " %d UUID %s", nodeId, gpuUuid);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

// detach a single GPU, and unregister GPU event
FMIntReturn_t
FMFabricConfig::configDetachGpu( uint32_t nodeId, char *gpuUuid )
{
    if ( !gpuUuid )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " %d GPU %s.", nodeId, gpuUuid ? gpuUuid : "Null UUID");
        return FM_INT_ST_ILWALID_GPU;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuDetachRequest *pDetachRequest = new lwswitch::gpuDetachRequest();

    pDetachRequest->set_partitionid( ILWALID_FABRIC_PARTITION_ID );

    //Fill in information, namely uuid of each gpu that needs to be detached
    lwswitch::gpuDetachRequestInfo *pDetachReqInfo = pDetachRequest->add_info();
    pDetachReqInfo->set_uuid( gpuUuid );
    pDetachReqInfo->set_unregisterevent( true );

    uint32_t physicalId;
    if ( mpGfm->getGpuPhysicalId(nodeId, gpuUuid, physicalId) == false )
    {
        // physicalId is no longer required as attach/detach is based on UUID
        FM_LOG_DEBUG("cannot find mapping for " NODE_ID_LOG_STR " %d GPU %s.", nodeId, gpuUuid);
    }
    else
    {
        pDetachReqInfo->set_gpuphysicalid(physicalId);
    }

    // send the final messages to LFM
    pMessage->set_type( lwswitch::FM_GPU_DETACH_REQ );
    pMessage->set_allocated_gpudetachreq( pDetachRequest );


    lwswitch::fmMessage *pResponse = NULL;

    rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                               &pResponse, mCfgMsgTimeoutSec );

    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU detach message to " NODE_ID_LOG_STR " %d failed with error %d",
                    nodeId, rc);
    }
    else if ( handleGpuDetachRespMsg( pResponse, false ) == false )
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to detach GPU " NODE_ID_LOG_STR " %d UUID %s", nodeId, gpuUuid);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
    }
    return rc;
}

// attach the GPUs in a specified partition synchronously
FMIntReturn_t
FMFabricConfig::configSharedLWSwitchPartitionAttachGPUs( uint32_t nodeId, PartitionInfo &partInfo )
{
    FMIntReturn_t rc;

    rc = attachGpus( nodeId, partInfo ) ;
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
    }
    return rc;
}

// detach the GPUs in a specified partition synchronously
FMIntReturn_t
FMFabricConfig::configSharedLWSwitchPartitionDetachGPUs( uint32_t nodeId, PartitionInfo &partInfo )
{
    FMIntReturn_t rc = detachGpus( nodeId, &partInfo ) ;
    if ( rc != FM_INT_ST_OK )
    {
        // error already logged
    }
    return rc;
}

// detach all GPUs in the node synchronously
FMIntReturn_t
FMFabricConfig::configDetachAllGPUs( uint32_t nodeId )
{
    return detachGpus( nodeId, NULL ) ;
}

FMIntReturn_t
FMFabricConfig::sendSwitchDisableLinkReq( uint32_t nodeId, uint32_t physicalId, uint64 disableMask )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::switchDisableLinkRequest *pDisableReqMsg = new lwswitch::switchDisableLinkRequest();

    // prepare the request message    
    pDisableReqMsg->set_switchphysicalid( physicalId );
    for ( uint32_t port = 0; port < MAX_PORTS_PER_LWSWITCH; port++ )
    {
        if ( (disableMask & ((uint64_t)1 << port)) )
        {
            pDisableReqMsg->add_portnum( port );
        }
    }

    pMessage->set_type( lwswitch::FM_SWITCH_DISABLE_LINK_REQ );
    pMessage->set_allocated_switchdisablelinkreq( pDisableReqMsg );

    rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true);
    if ( rc != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send LWSwitch link disable request to " NODE_ID_LOG_STR " %d failed with error %d", nodeId, rc);
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::configDisableSwitchTrunkLinks( uint32_t nodeId )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;

    // find the LWSwitch associated with specified node
    for ( it = mpGfm->mpParser->lwswitchCfg.begin();
          it != mpGfm->mpParser->lwswitchCfg.end(); it++ )
    {
        SwitchKeyType key = it->first;
        if ( key.nodeId == nodeId )
        {
            uint32_t physicalId = key.physicalId;
            uint64 trunkLinkMask;
            mpGfm->mpParser->getSwitchTrunkLinkMask( nodeId, physicalId, trunkLinkMask );
            FMPciInfo_t pciInfo = {0};
            mpGfm->getLWSwitchPciBdf(nodeId, physicalId, pciInfo);
            FM_LOG_INFO("disabling trunk links for LWSwitch " NODE_ID_LOG_STR " %d pci bus id %s physical id %d with link mask %#llx", 
                         nodeId, pciInfo.busId, physicalId, trunkLinkMask );
            rc = sendSwitchDisableLinkReq( nodeId, physicalId, trunkLinkMask );
            if ( rc != FM_INT_ST_OK )
            {
                // error already logged
                return rc;
            }
        }
    }

    return rc;
}

FMIntReturn_t
FMFabricConfig::SendMessageToLfm( uint32_t nodeId, uint32_t partitionId,
                                  lwswitch::fmMessage *pFmMessage, bool trackReq )
{
    FMIntReturn_t ret;
    FMIntReturn_t rc = FM_INT_ST_OK;

    if ( !pFmMessage )
    {
        FM_LOG_DEBUG("Invalid message to " NODE_ID_LOG_STR " %d, partitionId %d.",
                    nodeId, partitionId);
        return FM_INT_ST_MSG_SEND_ERR;
    }

    // add request to our context for tracking
    // before sending the message, add it to the list as the response can
    // come before even we add it to the list.
    int id = mpGfm->getControlMessageRequestId(nodeId);
    pFmMessage->set_requestid(id);
    if (trackReq) {
        FM_LOG_DEBUG("addPendingConfigRequest RequestId=%d type=%d", id, pFmMessage->type());
        addPendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    }

    ret = mpGfm->SendMessageToLfm(nodeId, pFmMessage, trackReq);
    if ( ret != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send socket message to local fabric manager for " NODE_ID_LOG_STR " %d failed with error %d", nodeId, ret);
        rc = FM_INT_ST_MSG_SEND_ERR;

        if ( trackReq )
            removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    }

    delete pFmMessage;
    return rc;
}

FMIntReturn_t
FMFabricConfig::SendMessageToLfmSync( uint32_t nodeId, uint32_t partitionId, lwswitch::fmMessage *pFmMessage,
                                      lwswitch::fmMessage **pResponse, uint32_t timeoutSec )
{
    FMIntReturn_t ret;
    FMIntReturn_t rc = FM_INT_ST_OK;

    if ( !pFmMessage )
    {
        FM_LOG_DEBUG("Invalid message to nodeId %d, partitionId %d.",
                    nodeId, partitionId);
        return FM_INT_ST_MSG_SEND_ERR;
    }

    int id = mpGfm->getControlMessageRequestId(nodeId);
    pFmMessage->set_requestid(id);

    ret = mpGfm->SendMessageToLfmSync(nodeId, pFmMessage, pResponse, timeoutSec);
    if ( ret != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("request to send sync message to local fabric manager for " NODE_ID_LOG_STR " %d failed with error %d", nodeId, ret);
        rc = FM_INT_ST_MSG_SEND_ERR;
    }

    delete pFmMessage;
    return rc;
}

void
FMFabricConfig::handleMessage( lwswitch::fmMessage  *pFmMessage )
{
    FM_LOG_DEBUG("message type %d", pFmMessage->type());

    switch ( pFmMessage->type() )
    {
    case lwswitch::FM_NODE_GLOBAL_CONFIG_RSP:
        dumpMessage(pFmMessage);
        handleNodeGlobalConfigRespMsg( pFmMessage );
        break;

    case lwswitch::FM_SWITCH_PORT_CONFIG_RSP:
        dumpMessage(pFmMessage);
        handleSWPortConfigRespMsg( pFmMessage, true );
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
        handleGpuConfigRespMsg( pFmMessage, true );
        break;

    case lwswitch::FM_GPU_ATTACH_RSP:
        dumpMessage(pFmMessage);
        handleGpuAttachRespMsg( pFmMessage, true );
        break;

    case lwswitch::FM_GPU_DETACH_RSP:
        dumpMessage(pFmMessage);
        handleGpuDetachRespMsg( pFmMessage, true );
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
        handleConfigGpuSetDisabledLinkMaskRespMsg( pFmMessage, true );
        break;

    case lwswitch::FM_GPU_GET_GFID_RSP:
    case lwswitch::FM_GPU_CFG_GFID_RSP:
        dumpMessage(pFmMessage);
        break;

    case lwswitch::FM_NODE_INFO_ACK:
        dumpMessage(pFmMessage);
        handleConfigNodeInfoAckMsg( pFmMessage );
        break;

    case lwswitch::FM_RMAP_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRmapTableConfigRespMsg( pFmMessage, true );
        break;

    case lwswitch::FM_RID_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRidTableConfigRespMsg( pFmMessage, true );
        break;

    case lwswitch::FM_RLAN_TABLE_RSP:
        dumpMessage(pFmMessage);
        handleRlanTableConfigRespMsg( pFmMessage, true );
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case lwswitch::FM_MCID_TABLE_SET_RSP:
        dumpMessage(pFmMessage);
        handleMulticastConfigRespMsg( pFmMessage, true );
        break;
#endif

    default:
        FM_LOG_ERROR("unknown message type %d received in fabric config handler", pFmMessage->type());
        break;
    }
}

void
FMFabricConfig::dumpMessage( lwswitch::fmMessage *pFmMessage )
{
#ifdef DEBUG
    std::string msgText;

    google::protobuf::TextFormat::PrintToString(*pFmMessage, &msgText);
    FM_LOG_DEBUG("%s", msgText.c_str());
#endif
}

/**
 *  on GFM, handle FM_NODE_GLOBAL_CONFIG_RSP message from LFM
 */
bool
FMFabricConfig::handleNodeGlobalConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_globalconfigresponse() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::nodeGlobalConfigResponse respMsg = pFmMessage->globalconfigresponse();
    uint32_t nodeId = pFmMessage->nodeid();

    bool ret = true;
    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        FM_LOG_ERROR("received global config response with error %d from " NODE_ID_LOG_STR " %d ", nodeId, respMsg.status());
        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_NODE_FAILED, errMsg );
        ret = false;
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
    return ret;
}

/**
 *  on GFM, handle FM_SWITCH_PORT_CONFIG_RSP message from LFM
 */
bool
FMFabricConfig::handleSWPortConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_portconfigresponse() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::switchPortConfigResponse &respMsg = pFmMessage->portconfigresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        return false;
    }

    bool ret = true;
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
            FM_LOG_ERROR("received switch port config response with error %d from " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);
            ret = false;

            if ( handleErr )
            {
                lwswitch::fmMessage errMsg = *pFmMessage;
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, handle FM_INGRESS_REQUEST_TABLE_RSP message from LFM
 */
bool
FMFabricConfig::handleIngReqTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_requesttableresponse() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::switchPortRequestTableResponse &respMsg = pFmMessage->requesttableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return false;
    }

    bool ret = true;
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
            ret = false;
            FM_LOG_ERROR("received ingress request config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, handle FM_INGRESS_RESPONSE_TABLE_RSP message from LFM
 */
bool
FMFabricConfig::handleIngRespTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_responsetableresponse() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::switchPortResponseTableResponse &respMsg = pFmMessage->responsetableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        return false;
    }

    bool ret = true;
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
            ret = false;
            FM_LOG_ERROR("received ingress response table config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
            }
            else
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, handle FM_GANGED_LINK_TABLE_RSP message from LFM
 */
bool
FMFabricConfig::handleGangedLinkTblConfigRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_gangedlinktableresponse() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::switchPortGangedLinkTableResponse &respMsg = pFmMessage->gangedlinktableresponse();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        return false;
    }

    bool ret = true;
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
            ret = false;
            FM_LOG_ERROR("received ganged link config response with error %d from " NODE_ID_LOG_STR " %d for LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);

            lwswitch::fmMessage errMsg = *pFmMessage;
            handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, common logic to handle FM_GPU_CONFIG_RSP,  FM_GPU_ATTACH_RSP, FM_GPU_DETACB_RSP message from LFM
 *
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender configGpu, the error response is
 *  handled by the sync message sender.
 */
bool
FMFabricConfig::handleCommonGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage,
                                              const lwswitch::gpuConfigResponse *respMsg,
                                              bool handleErr )
{
    if ( !pFmMessage || !respMsg )
    {
        FM_LOG_ERROR("Empty GPU config response.");
        return false;
    }

    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg->response_size() == 0 )
    {
        // no instance response, when all GPUs are attached/detached.
        FM_LOG_DEBUG("No instance response.");
        return true;
    }

    bool ret = true;
    for ( int i = 0; i < respMsg->response_size(); i++ )
    {
        const lwswitch::configResponse &instanceResponse = respMsg->response(i);

        if ( instanceResponse.has_partitionid() )
        {
            partitionId = instanceResponse.partitionid();
        }

        if ( instanceResponse.has_status() &&
             ( instanceResponse.status() != lwswitch::CONFIG_SUCCESS ) )
        {
            FM_LOG_ERROR("received GPU config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d GPU Id %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1);

            ret = false;

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                if ( handleErr )
                {
                    mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
                }
            }
            else
            {
                if ( handleErr )
                {
                    handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
                }
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, handle FM_GPU_CONFIG_RSP message from LFM
 *
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender configGpu, the error response is
 *  handled by the sync message sender.
 */
bool
FMFabricConfig::handleGpuConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{

    if ( !pFmMessage->has_gpuconfigrsp() )
    {
        FM_LOG_ERROR("Empty GPU config response.");
        return false;
    }

    const lwswitch::gpuConfigResponse *respMsg = (lwswitch::gpuConfigResponse *)&pFmMessage->gpuconfigrsp();
    return handleCommonGpuConfigRespMsg( pFmMessage, respMsg, handleErr);
}

/**
 *  on GFM, handle FM_GPU_ATTACH_RSP message from LFM
 *
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender configAttachGpu, the error response is
 *  handled by the sync message sender.
 */
bool
FMFabricConfig::handleGpuAttachRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_gpuattachrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty GPU attach response.");
        return false;
    }

    const lwswitch::gpuConfigResponse *respMsg = (lwswitch::gpuConfigResponse *)&pFmMessage->gpuattachrsp();
    return handleCommonGpuConfigRespMsg( pFmMessage, respMsg, handleErr);

}

/**
 *  on GFM, handle FM_GPU_DETACH_RSP message from LFM
 *
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender configDetachGpu, the error response is
 *  handled by the sync message sender.
 */
bool
FMFabricConfig::handleGpuDetachRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_gpudetachrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty GPU detach response.");
        return false;
    }

    const lwswitch::gpuConfigResponse *respMsg = (lwswitch::gpuConfigResponse *)&pFmMessage->gpudetachrsp();
    return handleCommonGpuConfigRespMsg( pFmMessage, respMsg, handleErr);
}
/**
 *  on GFM, handle FM_CONFIG_INIT_DONE_RSP sync response message from LFM
 */
bool
FMFabricConfig::handleConfigInitDoneRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_initdonersp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::configInitDoneRsp &respMsg = pFmMessage->initdonersp();
    uint32_t nodeId = pFmMessage->nodeid();
    bool ret = true;

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        ret = false;
        FM_LOG_ERROR("received init done config response with error %d from " NODE_ID_LOG_STR " %d", respMsg.status(), nodeId);

        // no need to handleConfigError,
        // as GFM will check this sync response, and abort on this error
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
    return ret;
}
/**
 *  on GFM, handle FM_NODE_INFO_ACK message from LFM
 */
bool
FMFabricConfig::handleConfigNodeInfoAckMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_nodeinfoack() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::nodeInfoAck &respMsg = pFmMessage->nodeinfoack();
    uint32_t nodeId = pFmMessage->nodeid();
    bool ret = true;

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        ret = false;
        FM_LOG_ERROR("received config info response with error %d from " NODE_ID_LOG_STR " %d", respMsg.status(), nodeId);
        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_NODE_FAILED, errMsg );
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
    return ret;
}


/**
 *  on GFM, handle FM_CONFIG_DEINIT_RSP message from LFM
 */
bool
FMFabricConfig::handleConfigDeinitRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_deinitrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::configDeInitRsp &respMsg = pFmMessage->deinitrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    bool ret = true;

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        ret = false;
        FM_LOG_ERROR("received deinit done response with error %d from " NODE_ID_LOG_STR " %d", respMsg.status(), nodeId);
        // for this deinit failure, don't generate our usual handleConfigError() path
        // as that will generate another config deinit request and potentially this
        // pattern will repeat.
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
    return ret;
}

/**
 *  on GFM, handle FM_SWITCH_DISABLE_LINK_RSP message from LFM
 */
bool
FMFabricConfig::handleConfigSwitchDisableLinkRespMsg( lwswitch::fmMessage *pFmMessage )
{
    if ( !pFmMessage->has_switchdisablelinkrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::switchDisableLinkResponse &respMsg = pFmMessage->switchdisablelinkrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    bool ret = true;

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        ret = false;
        FM_LOG_ERROR("received LWSwitch disable link config response with error %d from " NODE_ID_LOG_STR " %d", respMsg.status(), nodeId);

        lwswitch::fmMessage errMsg = *pFmMessage;
        handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_FAILED, errMsg );
    }

    FM_LOG_DEBUG("FM_SWITCH_DISABLE_LINK_RSP completed for " NODE_ID_LOG_STR " %d, requestId %d.",
                nodeId, pFmMessage->requestid());
    removePendingConfigRequest( nodeId, pFmMessage->requestid() );
    return ret;
}

/**
 *  on GFM, handle FM_GPU_SET_DISABLED_LINK_MASK_RSP message from LFM
 */
bool
FMFabricConfig::handleConfigGpuSetDisabledLinkMaskRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_gpusetdisabledlinkmaskrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::gpuSetDisabledLinkMaskResponse &respMsg = pFmMessage->gpusetdisabledlinkmaskrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    bool ret = true;

    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;
    if ( respMsg.has_partitionid() )
    {
        partitionId = respMsg.partitionid();
    }

    if ( respMsg.has_status() &&
         respMsg.status() != lwswitch::CONFIG_SUCCESS )
    {
        ret = false;
        // failure will have corresponding GPU uuid information
        std::string strUuid = "";
        if ( respMsg.has_uuid() )
        {
            strUuid = respMsg.uuid();
        }

        FM_LOG_ERROR("received GPU disable link config response with error %d from " NODE_ID_LOG_STR " %d for partidion id %d GPU Uuid %s",
                     respMsg.status(), nodeId, partitionId, strUuid.c_str() );

        lwswitch::fmMessage errMsg = *pFmMessage;
        if ( partitionId == ILWALID_FABRIC_PARTITION_ID )
        {
            if ( handleErr )
            {
                handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_GPU_FAILED, errMsg );
            }
        }
        else
        {
            if ( handleErr )
            {
                mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
            }
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/**
 *  on GFM, handle FM_GPU_GET_GFID_RSP message from LFM
 */
bool
FMFabricConfig::handleGetGfidRespMsg(lwswitch::fmMessage *pFmMessage, std::list<GpuGfidInfo> &gfidList)
{
    const lwswitch::gpuGetGfidResponse *respMsg;
    uint32_t partitionId;

    if (!pFmMessage->has_gpugetgfidrsp())
    {
        // empty response
        FM_LOG_DEBUG("empty GPU gfid response.");
        return false;
    }

    respMsg = (lwswitch::gpuGetGfidResponse *)&pFmMessage->gpugetgfidrsp();
    partitionId = respMsg->partitionid();

    if (respMsg->status() != lwswitch::CONFIG_SUCCESS) {
        FM_LOG_DEBUG("failed to get gfid for all GPUs in partition %d", partitionId);
        return false;
    }

    if (respMsg->info_size() == 0)
    {
        // no instance response, when GPUs are not configured with GFID.
        FM_LOG_DEBUG("no instance response.");
        return false;
    }

    for (int i = 0; i < respMsg->info_size(); i++)
    {
        const lwswitch::gpuGetGfidResponseInfo &instanceResponse = respMsg->info(i);
        GpuGfidInfo gfidInfo;

        // Update Gfid and GfidMask values for each GPU in the partition
        gfidInfo.gfid = instanceResponse.gfid();
        gfidInfo.gfidMask = instanceResponse.gfidmask();
        gfidList.push_back(gfidInfo);
    }

    return true;
}

/**
 *  on GFM, handle FM_GPU_CFG_GFID_RSP message from LFM
 */
bool
FMFabricConfig::handleCfgGfidRespMsg(lwswitch::fmMessage *pFmMessage)
{
    const lwswitch::gpuCfgGfidResponse *respMsg;
    uint32_t partitionId;

    if ( !pFmMessage->has_gpucfggfidrsp() )
    {
        // empty response
        FM_LOG_DEBUG("empty GPU gfid response.");
        return false;
    }

    respMsg = (lwswitch::gpuCfgGfidResponse *)&pFmMessage->gpucfggfidrsp();
    partitionId = respMsg->partitionid();

    if (respMsg->status() != lwswitch::CONFIG_SUCCESS) {
        FM_LOG_DEBUG("failed to configure gfid for all GPUs in partition %d", partitionId);
        return false;
    }

    return true;
}

void
FMFabricConfig::handleConfigError( uint32_t nodeId, GlobalFMErrorSource errSource,
                                   GlobalFMErrorTypes errType, lwswitch::fmMessage &errMsg )
{
    GlobalFMErrorHndlr errHndlr(mpGfm, nodeId, 0, errSource, errType, errMsg);
    errHndlr.processErrorMsg();
}

void
FMFabricConfig::handleSharedLWSwitchPartitionConfigError( uint32_t nodeId, uint32_t partitionId,
                                                          GlobalFMErrorSource errSource,
                                                          GlobalFMErrorTypes errType,
                                                          lwswitch::fmMessage &errMsg )
{
    GlobalFMErrorHndlr errHndlr(mpGfm, nodeId, partitionId, errSource, errType, errMsg);
    errHndlr.processErrorMsg();
}

void
FMFabricConfig::addPendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                         uint32_t partitionId  )
{
    FMAutoLock lock(mLock);

    std::map <ConfigRequestKeyType, unsigned int>::iterator it;
    ConfigRequestKeyType key;

    key.nodeId = nodeId;
    key.requestId = requestId;
    key.partitionId = partitionId;

    it = mPendingConfigReqMap.find( key );
    if ( it != mPendingConfigReqMap.end() )
    {
        FM_LOG_DEBUG("Request " NODE_ID_LOG_STR " %d, partitionId %d, requestId %d is already pending",
                    nodeId, partitionId, requestId);
        return;
    }

    mPendingConfigReqMap.insert(make_pair(key, requestId));
}

void
FMFabricConfig::removePendingConfigRequest( uint32_t nodeId, uint32_t requestId,
                                            uint32_t partitionId )
{
    FMAutoLock lock(mLock);

    std::map <ConfigRequestKeyType, unsigned int>::iterator it;
    ConfigRequestKeyType key;

    key.nodeId = nodeId;
    key.requestId = requestId;
    key.partitionId = partitionId;

    it = mPendingConfigReqMap.find( key );
    if ( it == mPendingConfigReqMap.end() )
    {
        FM_LOG_DEBUG("Request " NODE_ID_LOG_STR " %d, partitionId %d, requestId %d is not pending",
                    nodeId, partitionId, requestId);
        return;
    }

    mPendingConfigReqMap.erase( key );
}

void
FMFabricConfig::clearPartitionPendingConfigRequest( uint32_t nodeId, uint32_t partitionId )
{
    FMAutoLock lock(mLock);

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

bool FMFabricConfig::isPendingConfigReqEmpty( uint32_t nodeId, uint32_t partitionId )
{
    FMAutoLock lock(mLock);
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

void FMFabricConfig::dumpPendingConfigReq( uint32_t nodeId, uint32_t partitionId )
{
    FMAutoLock lock(mLock);
    std::map <ConfigRequestKeyType, unsigned int>::iterator it;

    FM_LOG_DEBUG("Pending Config requests on " NODE_ID_LOG_STR " %d", nodeId);
    for ( it = mPendingConfigReqMap.begin();
          it != mPendingConfigReqMap.end();
          it++ )
    {
        ConfigRequestKeyType key = it->first;
        if ( ( key.nodeId != nodeId ) || ( key.partitionId != partitionId ) )
             continue;

        FM_LOG_DEBUG("requestId: %d", it->second);
    }
}


/**
 *  on GFM, send routing and GPU configuration to LFM when a partition is activated or deactivated
 */
FMIntReturn_t
FMFabricConfig::configSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo,
                                               bool activate )
{
    FMIntReturn_t rc;

    // reconfigure all switch ports, as port config are all gone after switch port reset
    if ( activate )
    {
        PartitionSwitchInfoList::iterator it;
        for ( it = partInfo.switchInfo.begin(); it != partInfo.switchInfo.end(); it++ )
        {
            std::list<PortKeyType> portList;
            portList.clear();

            PartitionSwitchInfo switchInfo = *it;
            PortKeyType key;
            key.nodeId = nodeId;
            key.physicalId = switchInfo.physicalId;

            for (uint32_t portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
            {
                if ( ( switchInfo.enabledLinkMask & ( (uint64_t)1 << portIndex ) ) != 0 )
                {
                    key.portIndex = portIndex;
                    portList.push_back( key );
                }
            }

            if ( portList.size() > 0 )
            {
                rc = configSwitchPortList( nodeId, partInfo.partitionId, portList, false );
                if ( rc != FM_INT_ST_OK )
                {
                    return rc;
                }
            }
        }
    }

    // setup ingress request and response table
    rc = configSharedLWSwitchPartitionRoutingTable( nodeId, partInfo, activate );
    if ( rc != FM_INT_ST_OK )
    {
        return rc;
    }

    // configure/unconfigure the GPUs based on the activate flag
    if ((mpGfm->getFabricMode() == FM_MODE_VGPU) || (activate)) {
        rc = configSharedLWSwitchPartitionGPUs(nodeId, partInfo, activate);
        if ( rc != FM_INT_ST_OK )
        {
            return rc;
        }
    }

    // Make sure all config responses are back
    if ( waitForPartitionConfigCompletion(nodeId, partInfo.partitionId, PARTITION_CFG_TIMEOUT_MS) == false )
    {
        rc = FM_INT_ST_CFG_TIMEOUT;
    }

    return rc;
}

/**
 *  on GFM, Get GFID for all GPUs when a partition is being activated
 */
FMIntReturn_t
FMFabricConfig::configGetGfidForPartition(uint32_t nodeId, PartitionInfo &partInfo, fmPciDevice_t *vfList,
                                          std::list<GpuGfidInfo> &gfidList)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuGetGfidRequest *pRequest = new lwswitch::gpuGetGfidRequest();
    uint32_t i = 0;

    FM_LOG_DEBUG("configGetGfidForPartition: partition id %d", partInfo.partitionId);

    pMessage->set_type(lwswitch::FM_GPU_GET_GFID_REQ);
    pMessage->set_allocated_gpugetgfidreq( pRequest );
    pRequest->set_partitionid(partInfo.partitionId);

    // Fill in information for each gpu 
    PartitionGpuInfoList::iterator it;
    for (it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++, i++)
    {
        PartitionGpuInfo gpuInfo = *it;
        lwswitch::gpuGetGfidRequestInfo *pInfo = pRequest->add_info();
        lwswitch::devicePciInfo *vf = new lwswitch::devicePciInfo();
        pInfo->set_uuid( gpuInfo.uuid );
        pInfo->set_physicalid(gpuInfo.physicalId);
        pInfo->set_allocated_vf(vf);
        vf->set_domain(vfList[i].domain);
        vf->set_bus(vfList[i].bus);
        vf->set_device(vfList[i].device);
        vf->set_function(vfList[i].function);
        FM_LOG_DEBUG("configGetGfidForPartition: VF 0x%x.%x.%x.%x", vf->domain(), vf->bus(), vf->device(), vf->function());
    }

    // send the final messages to LFM
    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync(nodeId, partInfo.partitionId, pMessage, &pResponse, mCfgMsgTimeoutSec);
    if (rc != FM_INT_ST_OK || (pResponse == NULL))
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send get gfid message for " NODE_ID_LOG_STR " %d, partition id %d failed with error %d",
                     nodeId, partInfo.partitionId, rc);
    }
    else if (handleGetGfidRespMsg(pResponse, gfidList) == false)
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to get gfid for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partInfo.partitionId);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if (pResponse)
    {
        delete pResponse;
    }
    return rc;
}

/**
 *  on GFM, COnfigure GFID for all GPUs when a partition is being activated or deactivated
 */
FMIntReturn_t
FMFabricConfig::configCfgGfidForPartition(uint32_t nodeId, PartitionInfo &partInfo, bool activate)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    lwswitch::gpuCfgGfidRequest *pRequest = new lwswitch::gpuCfgGfidRequest();
    uint32_t i = 0;

    FM_LOG_DEBUG("configCfgGfidForPartition: activate %d " NODE_ID_LOG_STR " %d partition id %d", activate, nodeId, partInfo.partitionId);

    pMessage->set_type(lwswitch::FM_GPU_CFG_GFID_REQ);
    pMessage->set_allocated_gpucfggfidreq(pRequest);
    pRequest->set_partitionid(partInfo.partitionId);
    pRequest->set_activate(activate);

    // Fill in information for each gpu 
    PartitionGpuInfoList::iterator it;
    for (it = partInfo.gpuInfo.begin(); it != partInfo.gpuInfo.end(); it++, i++)
    {
        PartitionGpuInfo gpuInfo = *it;
        lwswitch::gpuCfgGfidRequestInfo *pInfo = pRequest->add_info();
        pInfo->set_uuid(gpuInfo.uuid);
        pInfo->set_physicalid(gpuInfo.physicalId);
        pInfo->set_gfid(gpuInfo.dynamicInfo.gfidInfo.gfid);
    }

    // send the final messages to LFM
    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync(nodeId, partInfo.partitionId, pMessage, &pResponse, mCfgMsgTimeoutSec);
    if (rc != FM_INT_ST_OK || (pResponse == NULL))
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send configure gfid message for " NODE_ID_LOG_STR " %d, partition id %d failed with error %d",
                     nodeId, partInfo.partitionId, rc);
    }
    else if (handleCfgGfidRespMsg(pResponse) == false)
    {
        // the response message carried error response
        FM_LOG_ERROR("failed to configure gfid for " NODE_ID_LOG_STR " %d partition id %d", nodeId, partInfo.partitionId);
        rc = FM_INT_ST_ILWALID_GPU_CFG;
    }

    if (pResponse)
    {
        delete pResponse;
    }
    return rc;
}

/**
 *  on GFM, send routing and GPU configurations to LFM when a partition is activated
 */
FMIntReturn_t
FMFabricConfig::configActivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo )
{
    return configSharedLWSwitchPartition( nodeId, partInfo, true );
}

/**
 *  on GFM, send routing and GPU configurations to LFM when a partition is deactivated
 */
FMIntReturn_t
FMFabricConfig::configDeactivateSharedLWSwitchPartition( uint32_t nodeId, PartitionInfo &partInfo )
{
    return configSharedLWSwitchPartition( nodeId, partInfo, false );
}

/**
 *  on GFM, send information to LFM about GPU LWLinks to be disabled for a partition
 */
FMIntReturn_t
FMFabricConfig::configSetGpuDisabledLinkMaskForPartition( uint32_t nodeId, PartitionInfo &partInfo )
{
    FMIntReturn_t rc;
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
        for ( uint32 idx = 0; idx < MAX_LWLINKS_PER_GPU; idx++ )
        {
            //
            // first check whether this link is actually visible to the driver
            // For Volta based GPUs, we should be setting mask for 6 links and 
            // for Ampere, it should be 12. This means we can't iterate for the MAX_LWLINKS as
            // that will compute disabled link mask for 12 Links in case of Volta.
            //
            if ((gpuInfo.discoveredLinkMask & (1 << idx)) == 0)
            {
                // skip as the specified link index is not even discovered in driver.
                continue;
            }

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

    lwswitch::fmMessage *pResponse = NULL;
    rc = SendMessageToLfmSync( nodeId, partInfo.partitionId, pMessage,
                               &pResponse, mCfgMsgTimeoutSec );
    if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
    {
        // failed to send the sync message
        FM_LOG_ERROR("request to send GPU disable link config message to " NODE_ID_LOG_STR " %d partition id %d failed with error %d",
                     nodeId, partInfo.partitionId, rc);
    }
    else if ( handleConfigGpuSetDisabledLinkMaskRespMsg( pResponse, false ) == false )
    {
        // error already logged
        rc = FM_INT_ST_ILWALID_PORT_CFG;
    }

    if ( pResponse )
    {
        delete pResponse;
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
FMFabricConfig::removeDisabledPortsFromRoutingEntry( uint32_t nodeId, uint32_t physicalId,
                                                     int32_t *vcmodevalid7_0, int32_t *vcmodevalid15_8,
                                                     int32_t *vcmodevalid17_16, int32_t *entryValid )
{
    uint64_t enabledLinkMask = 0;
    FMLWSwitchInfoMap::iterator it;
    FMLWSwitchInfoMap switchInfoMap = mpGfm->getLwSwitchInfoMap();

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        FMLWSwitchInfoList switchList = it->second;
        FMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            FMLWSwitchInfo switchInfo = (*jit);
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
FMFabricConfig::removeDisabledPortsFromIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
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
FMFabricConfig::removeDisabledPortsFromIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
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
FMFabricConfig::waitForPartitionConfigCompletion( uint32_t nodeId,
                                                  uint32_t partitionId,
                                                  uint32_t timeoutMs )
{
    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
    const unsigned int WAIT_MS = 50; // wait interval
    bool ret = true;

    while (true) {
        if ( isPendingConfigReqEmpty( nodeId, partitionId ) )
        {
            // no more pending requests
            break;
        }

        timeNow = timelib_usecSince1970();
        if ( (timeNow - timeStart) + WAIT_MS*1000 > timeoutMs*1000 ) {
             // elapsed all the time and there are still pending requests.
             // timed out
             FM_LOG_ERROR("request to configure fabric partition id %d has timed out", partitionId);
             ret = false;
             break;
        }
        // wait for more time
        usleep( WAIT_MS * 1000 );
    }

    if (mpGfm->mGfmPartitionMgr->isPartitionConfigFailed(nodeId, partitionId))
    {
        // error has oclwrred during the wait
        // error already logged when the error response is processed
        ret = false;
    }

    // Clear pending request before starting config this partition
    clearPartitionPendingConfigRequest( nodeId, partitionId );
    return ret;
}

/*
 * Evaluate the switch enabled link mask, remove disabled ports from rid port list.
 * If port list is empty, the entry is set to invalid
 */
void
FMFabricConfig::removeDisabledPortsFromRidEntry( uint32_t nodeId, uint32_t physicalId,
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
FMFabricConfig::removeDisabledPortsFromRlanEntry( uint32_t nodeId, uint32_t physicalId,
                                                  rlanRouteEntry *entry )
{
    // Limerock TODO
    return;
}

FMIntReturn_t
FMFabricConfig::configRmapTableEntriesByTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                               int portIndex, bool sync, RemapTable remapTable )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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

    for ( key.index = 0; key.index < FMDeviceProperty::getRemapTableSize(mpGfm->getSwitchArchType()); key.index++ )
    {
        pCfg = getRmapEntryByTable( key, remapTable );
        if ( !pCfg )
        {
            info = NULL;
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
            info->set_table( remapTable );
        }
        pMsg = info->add_entry();
        pMsg->CopyFrom( *pCfg );

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_entryvalid( 0 );
        }
        count++;

        FM_LOG_DEBUG("portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                     portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        if (sync)
        {
            lwswitch::fmMessage *pResponse = NULL;
            rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                                       &pResponse, mCfgMsgTimeoutSec );
            if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
            {
                // failed to send the sync message
                FM_LOG_ERROR("request to send route map config message to " NODE_ID_LOG_STR " %d failed with error %d",
                             nodeId, rc);
            }
            else if ( handleRmapTableConfigRespMsg( pResponse, false ) == false )
            {
                // the response message carried error response
                FM_LOG_ERROR("failed to config route map on " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d",
                             nodeId, pSwitchInfo->switchphysicalid(), portIndex);
                rc = FM_INT_ST_ILWALID_PORT_CFG;
            }

            if ( pResponse ) delete pResponse;
        }
        else
        {
            rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
            if ( rc != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("request to send route map config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                             nodeId, pSwitchInfo->switchphysicalid(), rc);
            }
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRmapEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   int portIndex, bool sync )
{
    FMIntReturn_t rc;

    rc = configRmapTableEntriesByTable( nodeId, pSwitchInfo, portIndex, sync, NORMAL_RANGE);
    if ( rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    rc = configRmapTableEntriesByTable( nodeId, pSwitchInfo, portIndex, sync, EXTENDED_RANGE_A);
    if ( rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }

    rc = configRmapTableEntriesByTable( nodeId, pSwitchInfo, portIndex, sync, EXTENDED_RANGE_B);
    if ( rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }

    rc = configRmapTableEntriesByTable( nodeId, pSwitchInfo, portIndex, sync, MULTICAST_RANGE);
    if ( rc != FM_INT_ST_OK )
    {
        // error is already logged
        return rc;
    }
#endif
    return rc;
}

FMIntReturn_t
FMFabricConfig::configRidEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                  int portIndex, bool sync)
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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

    for ( key.index = 0; key.index < FMDeviceProperty::getRidTableSize(mpGfm->getSwitchArchType()); key.index++ )
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
            FM_LOG_ERROR("invalid route id entries for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d index %d",
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

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_valid( 0 );
        }
        count++;

        FM_LOG_DEBUG("portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                    portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        if (sync)
        {
            lwswitch::fmMessage *pResponse = NULL;
            rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                                       &pResponse, mCfgMsgTimeoutSec );
            if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
            {
                // failed to send the sync message
                FM_LOG_ERROR("request to send route id config message to " NODE_ID_LOG_STR " %d failed with error %d",
                             nodeId, rc);
            }
            else if ( handleRidTableConfigRespMsg( pResponse, false ) == false )
            {
                // the response message carried error response
                FM_LOG_ERROR("failed to config route id on " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d",
                             nodeId, pSwitchInfo->switchphysicalid(), portIndex);
                rc = FM_INT_ST_ILWALID_PORT_CFG;
            }
            if ( pResponse ) delete pResponse;
        }
        else
        {
            rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
            if ( rc != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("request to send ingress route id config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                             nodeId, pSwitchInfo->switchphysicalid(), rc);
            }
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRlanEntries( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo,
                                   int portIndex, bool sync )
{
    FMIntReturn_t rc = FM_INT_ST_OK;
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

    for ( key.index = 0; key.index < FMDeviceProperty::getRlanTableSize(mpGfm->getSwitchArchType()); key.index++ )
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
            FM_LOG_ERROR("invalid route lan entries for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d index %d",
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

        if ((mpGfm->getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (mpGfm->getFabricMode() == FM_MODE_VGPU))
        {
            // shared fabric mode is on
            // set all request entries invalid
            // entries will be set to valid when the partition is activated
            pMsg->set_valid( 0 );
        }
        count++;

        FM_LOG_DEBUG("portIndex %d, firstIndex %d, key.index %d, pCfg->index %d, pMsg->index %d",
                     portIndex, info->firstindex(), key.index, pCfg->index(), pMsg->index());
    }

    if ( count > 0 )
    {
        if (sync)
        {
            lwswitch::fmMessage *pResponse = NULL;
            rc = SendMessageToLfmSync( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage,
                                       &pResponse, mCfgMsgTimeoutSec );
            if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
            {
                // failed to send the sync message
                FM_LOG_ERROR("request to send route lan config message to " NODE_ID_LOG_STR " %d failed with error %d",
                             nodeId, rc);
            }
            else if ( handleRlanTableConfigRespMsg( pResponse, false ) == false )
            {
                // the response message carried error response
                FM_LOG_ERROR("failed to config route lan on " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d",
                             nodeId, pSwitchInfo->switchphysicalid(), portIndex);
                rc = FM_INT_ST_ILWALID_PORT_CFG;
            }
            if ( pResponse ) delete pResponse;
        }
        else
        {
            rc = SendMessageToLfm( nodeId, ILWALID_FABRIC_PARTITION_ID, pMessage, true );
            if ( rc != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("request to send ingress route lan config to " NODE_ID_LOG_STR " %d for LWSwitch physical id %d failed with error %d",
                             nodeId, pSwitchInfo->switchphysicalid(), rc);
            }
        }
    }
    else
    {
        delete pMessage;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d, portIndex %d, count %d, rc %d",
                nodeId, pSwitchInfo->switchphysicalid(), portIndex, count, rc);

    return rc;
}

FMIntReturn_t
FMFabricConfig::configRmapTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRmapEntries( nodeId, pSwitchInfo, key.portIndex, false );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configRidTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRidEntries( nodeId, pSwitchInfo, key.portIndex, false );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricConfig::configRlanTable( uint32_t nodeId, lwswitch::switchInfo* pSwitchInfo )
{
    PortKeyType key;
    map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    FMIntReturn_t rc;

    key.nodeId  = nodeId;
    key.physicalId = pSwitchInfo->switchphysicalid();

    for ( key.portIndex = 0; key.portIndex <  MAX_PORTS_PER_LWSWITCH; key.portIndex++ )
    {
        it = mpGfm->mpParser->portInfo.find(key);
        if ( it == mpGfm->mpParser->portInfo.end() )
        {
            continue;
        }

        rc = configRlanEntries( nodeId, pSwitchInfo, key.portIndex, false );
        if ( rc != FM_INT_ST_OK )
        {
            // error already logged
            return rc;
        }
    }

    return FM_INT_ST_OK;
}


/*
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender, the error response is handled by
 *  the sync message sender.
*/
bool
FMFabricConfig::handleRmapTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_rmaptablersp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::portRmapTableResponse &respMsg = pFmMessage->rmaptablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return false;
    }

    bool ret = true;
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
            FM_LOG_ERROR("received route map config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);
            ret = false;

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                if ( handleErr )
                {
                    mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
                }
            }
            else
            {
                if ( handleErr )
                {
                    handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
                }
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/*
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender, the error response is handled by
 *  the sync message sender.
*/
bool
FMFabricConfig::handleRidTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_ridtablersp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::portRidTableResponse &respMsg = pFmMessage->ridtablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return false;
    }

    bool ret = true;
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
            FM_LOG_ERROR("received route id config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);
            ret = false;

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                if ( handleErr )
                {
                    mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
                }
            }
            else
            {
                if ( handleErr )
                {
                    handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
                }
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/*
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender, the error response is handled by
 *  the sync message sender.
*/
bool
FMFabricConfig::handleRlanTableConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_rlantablersp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::portRlanTableResponse &respMsg = pFmMessage->rlantablersp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return false;
    }

    bool ret = true;
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
            FM_LOG_ERROR("received rlan table config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);
            ret = false;

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                if ( handleErr )
                {
                    mpGfm->mGfmPartitionMgr->setPartitionConfigFailure(nodeId, partitionId);
                }
            }
            else
            {
                if ( handleErr )
                {
                    handleConfigError( nodeId, ERROR_SORUCE_SW_LOCALFM, ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED, errMsg );
                }
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}

/*
 *  This can be called from aync message handler FMFabricConfig::handleMessage, the error response
 *  is handled by the global config error handler.
 *
 *  This can be also be called from sync message sender, the error response is handled by
 *  the sync message sender.
*/
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
bool
FMFabricConfig::handleMulticastConfigRespMsg( lwswitch::fmMessage *pFmMessage, bool handleErr )
{
    if ( !pFmMessage->has_mcidtablesetrsp() )
    {
        // empty response
        FM_LOG_DEBUG("Empty response.");
        return false;
    }

    const lwswitch::mcidTableSetResponse &respMsg = pFmMessage->mcidtablesetrsp();
    uint32_t nodeId = pFmMessage->nodeid();
    uint32_t partitionId = ILWALID_FABRIC_PARTITION_ID;

    if ( respMsg.response_size() == 0 )
    {
        // no instance response
        FM_LOG_DEBUG("No instance response.");
        removePendingConfigRequest( nodeId, pFmMessage->requestid() );
        return false;
    }

    bool ret = true;
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
            FM_LOG_ERROR("received multicast table config response with error %d from " NODE_ID_LOG_STR " %d for partition id %d LWSwitch physical id %d port %d",
                        instanceResponse.status(), nodeId,
                        instanceResponse.has_partitionid() ? instanceResponse.partitionid() : -1,
                        instanceResponse.has_devicephysicalid() ? instanceResponse.devicephysicalid() : -1,
                        instanceResponse.has_port() ? instanceResponse.port() : -1);
            ret = false;

            lwswitch::fmMessage errMsg = *pFmMessage;
            if ( instanceResponse.has_partitionid() &&
                 (instanceResponse.partitionid() != ILWALID_FABRIC_PARTITION_ID) )
            {
                // handle partition config error
                if ( handleErr )
                {
                    // Laguna TODO
                    // handle multicast config error in a partition
                }
            }
            else
            {
                if ( handleErr )
                {
                    // Laguna TODO
                    // handle multicast config error in baremetal
                }
            }
            break;
        }
    }

    removePendingConfigRequest( nodeId, pFmMessage->requestid(), partitionId );
    return ret;
}
#endif

rmapPolicyEntry *
FMFabricConfig::getRmapEntryByTable( RmapTableKeyType key, RemapTable rmapTable )
{
    rmapPolicyEntry *entry = NULL;
    std::map <RmapTableKeyType, rmapPolicyEntry *>::iterator it;

    switch ( rmapTable )
    {
    case NORMAL_RANGE:
        it = mpGfm->mpParser->rmapEntry.find(key);
        if ( it != mpGfm->mpParser->rmapEntry.end() )
        {
            entry = it->second;
        }
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

    case EXTENDED_RANGE_A:
        it = mpGfm->mpParser->rmapExtAEntry.find(key);
        if ( it != mpGfm->mpParser->rmapExtAEntry.end() )
        {
            entry = it->second;
        }
        break;

    case EXTENDED_RANGE_B:
        it = mpGfm->mpParser->rmapExtBEntry.find(key);
        if ( it != mpGfm->mpParser->rmapExtBEntry.end() )
        {
            entry = it->second;
        }
        break;

    case MULTICAST_RANGE:
        it = mpGfm->mpParser->rmapMcEntry.find(key);
        if ( it != mpGfm->mpParser->rmapMcEntry.end() )
        {
            entry = it->second;
        }
        break;
#endif

    default:
        break;
    }

    return entry;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
// Send multicast route config message to LFM sync
FMIntReturn_t
FMFabricConfig::configMulticastRoutes( uint32_t nodeId, uint32_t switchPhysicalId,
                                       MulticastGroupInfo *groupInfo, PortMulticastTable &portMcastTable,
                                       bool freeGroup, bool sync )
{
    if ( !groupInfo )
    {
        return FM_INT_ST_BADPARAM;
    }

    FMIntReturn_t rc;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_MCID_TABLE_SET_REQ );

    lwswitch::mcidTableSetRequest *pConfigRequest = new lwswitch::mcidTableSetRequest;
    pMessage->set_allocated_mcidtablesetreq( pConfigRequest );

    pConfigRequest->set_switchphysicalid(switchPhysicalId);
    lwswitch::MulticastGroupInfo *pGroupInfo = pConfigRequest->add_groupinfo();

    pGroupInfo->set_partitionid(groupInfo->partitionId);
    pGroupInfo->set_mcid(groupInfo->mcId);

    lwswitch::MulticastMemoryMode reflectiveMode = groupInfo->reflectiveMode ?
            lwswitch::MC_MEMORY_MODE_REFLECTIVE : lwswitch::MC_MEMORY_MODE_MULTICAST;
    pGroupInfo->set_mode(reflectiveMode);
    pGroupInfo->set_excludeself(groupInfo->excludeSelf);
    pGroupInfo->set_nodynrsp(groupInfo->noDynRsp);

    // iterate through all ilwolved all ports on this switch
    lwswitch::SwitchMultcastTableInfo *pSwitchTable = new lwswitch::SwitchMultcastTableInfo;
    pGroupInfo->set_allocated_switchmulticasttable(pSwitchTable);

    pSwitchTable->set_nodeid(nodeId);
    pSwitchTable->set_switchphysicalid(switchPhysicalId);

    PortMulticastTable::iterator pit;
    for ( pit = portMcastTable.begin(); pit != portMcastTable.end(); pit++ ) {
        PortMulticastInfo &portInfo = pit->second;

        lwswitch::PortMulticastTableInfo *pPortTable = pSwitchTable->add_portmulticasttable();
        pPortTable->set_portnum(portInfo.portNum);

        MulticastSprays::iterator sprayIt;
        MulticastPorts::iterator portIt;
        PortMcIdTableEntry *entry;
        lwswitch::MulticastPortList *pPortList;
        lwswitch::MulticastPort *pPort;

        // mcId table entry
        lwswitch::MulticastTableEntry *pMcIdEntry = new lwswitch::MulticastTableEntry;
        pPortTable->set_allocated_tableentry(pMcIdEntry);

        entry = &portInfo.mcIdEntry;
        pMcIdEntry->set_index(entry->index);
        pMcIdEntry->set_extendedtbl(false);
        pMcIdEntry->set_entryvalid(!freeGroup);

        for ( sprayIt = entry->sprays.begin(); sprayIt != entry->sprays.end(); sprayIt++ ) {
            MulticastPortList &portList = *sprayIt;

            pPortList = pMcIdEntry->add_sprays();
            pPortList->set_replicavalid(portList.replicaValid);
            pPortList->set_replicaoffset(portList.replicaOffset);

            for ( portIt = portList.ports.begin(); portIt != portList.ports.end(); portIt++ ) {
                MulticastPort &port = *portIt;

                pPort = pPortList->add_ports();
                pPort->set_portnum(port.portNum);
                pPort->set_vchop(port.vcHop);
            }
        }

        // extended mcId table entry
        entry = &portInfo.extendedMcIdEntry;
        if (entry->extendedTable) {
            lwswitch::MulticastTableEntry *pExtMcIdEntry = new lwswitch::MulticastTableEntry;
            pPortTable->set_allocated_extendedtableentry(pExtMcIdEntry);

            pMcIdEntry->set_index(entry->index);
            pMcIdEntry->set_extendedtbl(true);
            pMcIdEntry->set_entryvalid(!freeGroup);

            for ( sprayIt = entry->sprays.begin(); sprayIt != entry->sprays.end(); sprayIt++ ) {
                MulticastPortList &portList = *sprayIt;

                pPortList = pMcIdEntry->add_sprays();
                pPortList->set_replicavalid(portList.replicaValid);
                pPortList->set_replicaoffset(portList.replicaOffset);

                for ( portIt = portList.ports.begin(); portIt != portList.ports.end(); portIt++ ) {
                    MulticastPort &port = *portIt;

                    pPort = pPortList->add_ports();
                    pPort->set_portnum(port.portNum);
                    pPort->set_vchop(port.vcHop);
                }
            }
        }
    }

    if ( sync )
    {
        lwswitch::fmMessage *pResponse = NULL;
        rc = SendMessageToLfmSync( nodeId, groupInfo->partitionId, pMessage,
                                   &pResponse, mCfgMsgTimeoutSec );
        if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
        {
            // failed to send the sync message
            FM_LOG_ERROR("request to send multicast config message to " NODE_ID_LOG_STR " %d failed with error %d",
                         nodeId, rc);
        }
        else if ( handleMulticastConfigRespMsg( pResponse, false ) == false )
        {
            // the response message carried error response
            FM_LOG_ERROR("failed to config multicast on " NODE_ID_LOG_STR " %d partitionId %d",
                         nodeId, groupInfo->partitionId);
            rc = FM_INT_ST_CFG_ERROR;
        }
        if ( pResponse ) delete pResponse;
    }
    else
    {
        rc = SendMessageToLfm( nodeId, groupInfo->partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send multicast config to " NODE_ID_LOG_STR " %d partition %d with error %d",
                         nodeId, groupInfo->partitionId, rc);
        }
    }

    return rc;

}

// Send multicast route config message to LFM sync
FMIntReturn_t
FMFabricConfig::configMulticastRemapTable( uint32_t nodeId, uint32_t switchPhysicalId,
                                           MulticastGroupInfo *groupInfo, PortMulticastTable &portMcastTable,
                                           uint64_t mappedAddr, bool freeGroup, bool sync )
{
    if ( !groupInfo )
    {
        return FM_INT_ST_BADPARAM;
    }

    FMIntReturn_t rc = FM_INT_ST_OK;
    lwswitch::fmMessage *pMessage = new lwswitch::fmMessage();
    pMessage->set_type(lwswitch::FM_RMAP_TABLE_REQ);

    lwswitch::portRmapTableRequest *pConfigRequest = new lwswitch::portRmapTableRequest;
    pMessage->set_allocated_rmaptablereq(pConfigRequest);
    pConfigRequest->set_switchphysicalid(switchPhysicalId);

    // program remap entry on all ilwolved all ports on this switch
    PortMulticastTable::iterator pit;
    for ( pit = portMcastTable.begin(); pit != portMcastTable.end(); pit++ )
    {
        PortMulticastInfo &portInfo = pit->second;

        lwswitch::portRmapTableInfo *info = pConfigRequest->add_info();
        info->set_switchphysicalid( switchPhysicalId );
        info->set_port( portInfo.portNum );

        // mcid is the index to multicast remap table
        info->set_firstindex( groupInfo->mcId );
        info->set_table( MULTICAST_RANGE );
        info->set_partitionid( groupInfo->partitionId );

        rmapPolicyEntry *entry = info->add_entry();
        entry->set_version( FABRIC_MANAGER_VERSION );
        entry->set_index( groupInfo->mcId );
        entry->set_entryvalid( freeGroup ? 0 : 1 );
        entry->set_address( mappedAddr );
        entry->set_targetid( groupInfo->mcId ); // targetId is mcId for multicast remap table

        // remapFlags = remap FLA multicast address, AddrType[1:0] 2’b10 Map Slot
        uint32_t  remapFlags = REMAP_FLAGS_REMAP_ADDR;
        if ( groupInfo->reflectiveMode == lwswitch::MC_MEMORY_MODE_REFLECTIVE )
        {
            remapFlags |= REMAP_FLAGS_REFLECTIVE;
        }
        entry->set_remapflags( remapFlags );
    }

    if ( sync )
    {
        lwswitch::fmMessage *pResponse = NULL;
        rc = SendMessageToLfmSync( nodeId, groupInfo->partitionId, pMessage,
                                   &pResponse, FM_CONFIG_MSG_TIMEOUT );
        if ( rc != FM_INT_ST_OK || ( pResponse == NULL ) )
        {
            // failed to send the sync message
            FM_LOG_ERROR("request to send multicast remap message to " NODE_ID_LOG_STR " %d failed with error %d",
                         nodeId, rc);
        }
        else if ( handleRmapTableConfigRespMsg( pResponse, false ) == false )
        {
            // the response message carried error response
            FM_LOG_ERROR("failed to map multicast address on " NODE_ID_LOG_STR " %d partition id %d",
                         nodeId, groupInfo->partitionId);
            rc = FM_INT_ST_CFG_ERROR;
        }
        if ( pResponse ) delete pResponse;
    }
    else
    {
        rc = SendMessageToLfm( nodeId, groupInfo->partitionId, pMessage, true );
        if ( rc != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("request to send multicast remap to " NODE_ID_LOG_STR " %d partition id %d with error %d",
                         nodeId, groupInfo->partitionId, rc);
        }
    }

    return rc;
}

#endif
