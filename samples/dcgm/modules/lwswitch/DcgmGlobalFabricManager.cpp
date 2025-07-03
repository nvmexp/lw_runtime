
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <sstream>
#include "DcgmModuleLwSwitch.h"
#include "dcgm_lwswitch_structs.h"
#include "topology.pb.h"
#include "dcgm_structs.h"
#include "lwcm_util.h"
#include "LwcmSettings.h"
#include "DcgmGlobalFabricManager.h"
#include "LwcmRequest.h"
#include "DcgmGFMHelper.h"
#include "logging.h"
#include "DcgmGlobalControlMsgHndl.h"
#include "DcgmGlobalStatsMsgHndlr.h"
#include "DcgmFMTopologyValidator.h"
#include "DcgmGlobalCommandServer.h"
#include "DcgmGlobalFMErrorHndlr.h"
#include "DcgmGFMFabricPartitionMgr.h"
#include "DcgmGlobalFabricManagerHaMgr.h"
#include "DcgmLogging.h"
#include <g_lwconfig.h>


DcgmGlobalFabricManager::DcgmGlobalFabricManager(bool sharedFabric,
                                                 bool restart,
                                                 unsigned short startingPort,
                                                 char *domainSocketPath,
                                                 char *stateFilename)
{
    FM_ERROR_CODE rc;


    if ((sharedFabric == false) && (restart == true))
    {
        PRINT_ERROR("", "Failed to restart Fabric Manager. Restart is only supported in shared LWSwitch multitenancy mode.");
        throw std::runtime_error("Failed to restart Fabric Manager. Restart is only supported in shared LWSwitch multitenancy mode.");
    }

    mSharedFabric = sharedFabric;
    mGpuInfoMap.clear();
    mBlacklistGpuInfoMap.clear();
    mLwswitchInfoMap.clear();
    mGfmPartitionMgr = NULL;
    mStartingPort = startingPort;
    mFabricManagerInitDone = false;
    mRestart = restart;
    mpHaMgr = NULL;

    lwosInitializeCriticalSection( &mLock );

    // Get the host engine and cache manager pointers so we can add our lwswitches to the cache manager
    mpHostEngineHandler = LwcmHostEngineHandler::Instance();
    if (NULL == mpHostEngineHandler)
    {
        PRINT_ERROR("", "DcgmGlobalFabricManager: failed to get Lwcm HostEngineHandler instance");
        throw std::runtime_error("DcgmGlobalFabricManager: failed to get LwcmHostEngineHandler instance");
    }

    mpCacheManager = mpHostEngineHandler->GetCacheManager();
    if (NULL == mpCacheManager)
    {
        PRINT_ERROR("", "DcgmGlobalFabricManager: failed to get DCGM Cache Manager instance");
        throw std::runtime_error("DcgmGlobalFabricManager: failed to get DCGM Cache Manager instance");
    }

    // initialize error handling work queue with one worker thread
    if (workqueue_init(&mErrWorkQueue, 1)) {
        PRINT_ERROR("", "failed to create global fabric manager error work queue");
        throw std::runtime_error("failed to create global fabric manager error work queue");
    }

    // create all the message handlers
    mpConfig = new DcgmFabricConfig( this );
    mMsgHandlers.push_back(mpConfig);

    mLinkTrainIntf = new DcgmFMLWLinkIntf(this);
    mMsgHandlers.push_back(mLinkTrainIntf);

    mDevInfoMsgHndlr = new GFMDevInfoMsgHdlr(this);
    mMsgHandlers.push_back(mDevInfoMsgHndlr);

    mControlMsgHndl = new DcgmGlobalControlMsgHndl(this);
    mMsgHandlers.push_back(mControlMsgHndl);

    mGlobalStatsMsgHndl = new DcgmGlobalStatsMsgHndlr(this);
    mMsgHandlers.push_back(mGlobalStatsMsgHndl);

#ifdef DEBUG
    // start our localFM command server interface
    mGlobalCmdServer = new DcgmGlobalCommandServer(this);
#endif

    // Parse from the topology
    mpParser = new DcgmFabricParser();
    rc = parseFabric();
    if ( rc != FM_SUCCESS ) {
        PRINT_ERROR("", "failed to open/parse fabric topology file information");
        throw std::runtime_error("failed to open/parse fabric topology file information");
    }

    // create our topology validator instance
    mTopoValidator = new DcgmFMTopologyValidator(this);

    // Start control connection to Local Fabric Managers
    if ( (mpParser->NodeCfg.size() > 1) && (domainSocketPath != NULL) )
    {
         PRINT_ERROR("", "Unix domain socket is not supported for multi-node configuration..");
         throw std::runtime_error("Unix domain socket is not supported for multi-node configuration.");
    }
    createFabricNodes(domainSocketPath);

    // wait for the FabricNode's control connections to establish
    if (!waitForAllFabricNodeCtrlConns()) {
        PRINT_ERROR("", "Failed to establish control connection with all the local fabric managers");
        throw std::runtime_error("Failed to establish control connection with all the local fabric managers");
    }

    // send all the general (non-switch/gpu) config to all nodes
    sendGlobalConfigToAllFabricNodes();

    // gather all GPU and LWSwitch device info, including physicalIds from all nodes.
    // get LWLink device information from CoreLib driver after disabling any links (like trunk links in multi-host)
    DcgmGFMHelper::getLWSwitchDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLwswitchInfoMap);
    DcgmGFMHelper::getGpuDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mGpuInfoMap, mBlacklistGpuInfoMap);
    dumpAllGpuInfo();
    dumpAllLWSwitchInfo();

    // FM has nothing to do in Non-LWSwitch environment. But, lw-hostengine (DCGM) should
    // keep running for GPU management related operations, so simply return
    DcgmFMLWSwitchInfoMap::iterator it;
    for (it = mLwswitchInfoMap.begin(); it != mLwswitchInfoMap.end(); it++) {
        if(it->second.size() != 0) {
            //if any node has LWSwitches GFM should run to manage a multi-node system
            break;
        }
    }
    if(it == mLwswitchInfoMap.end()) {
        FM_SYSLOG_NOTICE("No LWSwitches detected, skipping Fabric Manager specific initialization");
        printf("No LWSwitches detected, skipping Fabric Manager specific initialization\n");
        return;
    }

    // create fabric partition manager
    if ( mSharedFabric == true ) {
        mGfmPartitionMgr = new DcgmGFMFabricPartitionMgr(this);
    }

    // Create HA manager
    mpHaMgr = new DcgmGlobalFabricManagerHaMgr(mSharedFabric, stateFilename, this);

    if ( ( isRestart() == true ) &&
         ( mpHaMgr->validateAndLoadState() == false ) ) {
        std::ostringstream ss;
        ss << "Failed to validate and load restart states." << endl;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // done with all the common initialization sequence. branch based on GlobalFM mode
    if ( mSharedFabric == false ) {
        startNonSharedFabricMode();
    } else {
        if ( isRestart() == true ) {
            restartSharedFabricMode();
        } else {
            startSharedFabricMode();
        }

        // at this stage, some of the cached GPU related data is stale as the GPUs are detached
        // from ServiceVM. So, clearing them to make it explicit.
        mGpuInfoMap.clear();
        mBlacklistGpuInfoMap.clear();
        mGpuLWLinkConnMatrix.clear();
    }

    // wait and check if config error has oclwrred due to
    // error response or timeout
    if ( waitForAllFabricNodeConfigCompletion() != FM_SUCCESS )
    {
        std::ostringstream ss;
        ss << "Failed to configure all the available GPUs or LWSwitches." << endl;
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // All our initialization for the node is done. Tell LFM to allow peer access
    // by setting appropriate RM/LWSwitch driver state.
    sendInitDoneToAllFabricNodes();

    FM_SYSLOG_NOTICE("Successfully configured all the available GPUs and LWSwitches.");
    printf("Successfully configured all the available GPUs and LWSwitches.\n");

    // set shared fabric mode initialization status, to support DCGM FM APIs.
    mFabricManagerInitDone = true;

    // save the states if it is not a restart
    if ( isRestart() == false ) {
        mpHaMgr->saveStates();
    }
};

/*****************************************************************************/
DcgmGlobalFabricManager::~DcgmGlobalFabricManager()
{
    // TODO - move this to appropriate place (like error handling) later
    // Tell LFM to stop peer access by setting appropriate RM/LWSwitch driver state.
    sendDeInitToAllFabricNodes();

    workqueue_shutdown(&mErrWorkQueue);

#ifdef DEBUG
    delete mGlobalCmdServer;
#endif

    MsgHandlerList::iterator mit = mMsgHandlers.begin();
    while ( mit != mMsgHandlers.end() ) {
        FMMessageHandler* pMsgHandlr = *mit;
        mit = mMsgHandlers.erase( mit );
        delete pMsgHandlr;
    }

    std::map <uint32_t, DcgmFabricNode*>::iterator it = mvFabricNodes.begin();
    while ( it != mvFabricNodes.end() ) {
        DcgmFabricNode* pFabricNode = it->second;
        mvFabricNodes.erase( it++ );
        delete pFabricNode;
    }

    if (mTopoValidator) delete mTopoValidator;
    if (mpParser) delete mpParser;
    if (mGfmPartitionMgr) delete mGfmPartitionMgr;
    if (mpHaMgr) delete mpHaMgr;

    lwosDeleteCriticalSection( &mLock );
};

/*****************************************************************************/

int
DcgmGlobalFabricManager::ProcessMessage(uint32 nodeId, lwswitch::fmMessage * pFmMessage, bool &isResponse)
{
    // a locking is required as this can be called simultaneously by two nodes
    // as each socket run on its own thread.

    // TODO - Remove this locking once all the message handlers have their own locks.
    lwosEnterCriticalSection( &mLock );

    switch ( pFmMessage->type() ) {
        case lwswitch::FM_LWLINK_TRAIN_RSP_COMPLETE:
            mLinkTrainIntf->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_NODE_GET_LWLINK_DEVICE_INFO_RSP:
        case lwswitch::FM_NODE_GET_LWSWITCH_DEVICE_INFO_RSP:
        case lwswitch::FM_NODE_GET_GPU_DEVICE_INFO_RSP:            
            mDevInfoMsgHndlr->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_NODE_GLOBAL_CONFIG_RSP:
        case lwswitch::FM_SWITCH_PORT_CONFIG_RSP:
        case lwswitch::FM_INGRESS_REQUEST_TABLE_RSP:
        case lwswitch::FM_INGRESS_RESPONSE_TABLE_RSP:
        case lwswitch::FM_GPU_CONFIG_RSP:
        case lwswitch::FM_GANGED_LINK_TABLE_RSP:
        case lwswitch::FM_CONFIG_INIT_DONE_RSP:
        case lwswitch::FM_CONFIG_DEINIT_RSP:
        case lwswitch::FM_GPU_ATTACH_RSP:
        case lwswitch::FM_GPU_DETACH_RSP:
        case lwswitch::FM_SWITCH_DISABLE_LINK_RSP:
        case lwswitch::FM_GPU_SET_DISABLED_LINK_MASK_RSP:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
        case lwswitch::FM_RMAP_TABLE_RSP:
        case lwswitch::FM_RID_TABLE_RSP:
        case lwswitch::FM_RLAN_TABLE_RSP:
#endif
            mpConfig->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_HEARTBEAT_ACK:
            mControlMsgHndl->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_LWSWITCH_ERROR_FATAL:
        case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL:
        case lwswitch::FM_NODE_STATS_REPORT:
        case lwswitch::FM_NODE_STATS_ACK:
        case lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY:
        case lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY:
        case lwswitch::FM_LWLINK_ERROR_GPU_FATAL:
            mGlobalStatsMsgHndl->handleMessage( pFmMessage );
            isResponse = false;
            break;
        case lwswitch::FM_NODE_INFO_ACK:
            mpConfig->handleMessage( pFmMessage );
            isResponse = true;
            break;
        default:
            PRINT_ERROR("%d", "unknown message type %d", pFmMessage->type());
            isResponse = false;
            break;
    }

    lwosLeaveCriticalSection( &mLock );

    return FM_SUCCESS;
}

void
DcgmGlobalFabricManager::OnFabricNodeConnect(uint32 nodeId)
{
    MsgHandlerList::iterator it;
    for( it = mMsgHandlers.begin(); it != mMsgHandlers.end(); ++it ) {
        FMMessageHandler* msgHdlr = (*it);
        msgHdlr->handleEvent( FM_EVENT_PEER_FM_CONNECT, nodeId );
    }
}

void
DcgmGlobalFabricManager::OnFabricNodeDisconnect(uint32 nodeId)
{
    MsgHandlerList::iterator it;
    for( it = mMsgHandlers.begin(); it != mMsgHandlers.end(); ++it ) {
        FMMessageHandler* msgHdlr = (*it);
        msgHdlr->handleEvent( FM_EVENT_PEER_FM_DISCONNECT, nodeId );
    }

    // For single node systems, we can continue running even if there is a socket disconnection.
    // Connection retry logic will attempt to reconnect with localFM for every 1 second interval.
    // Lwrrently there is no re-configuration of LWSwitches/GPUs after a re-connection.
    // Note: we need to re-visit this for multi-node systems

    FM_SYSLOG_NOTICE("lost socket connection between global and local fabric manager instance");
}

dcgmReturn_t
DcgmGlobalFabricManager::SendMessageToGfm(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    PRINT_ERROR("", "SendMessageToGfm within Global Fabric Manager is not supposed to be called\n");
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t
DcgmGlobalFabricManager::SendMessageToLfm(uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage,
                                          bool trackReq)
{
    PRINT_DEBUG("","DcgmGlobalFabricManager: SendMessageToLfm ");
    std::map <uint32_t, DcgmFabricNode *>::iterator it;

    it = mvFabricNodes.find( fabricNodeId );
    if ( it == mvFabricNodes.end() )
    {
        PRINT_DEBUG("%d", "Invalid fabric node ID %d", fabricNodeId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    DcgmFabricNode *pFabricNode = it->second;
    return pFabricNode->SendControlMessage( pFmMessage, trackReq );
}

FM_ERROR_CODE
DcgmGlobalFabricManager::parseFabric()
{
    FM_ERROR_CODE rc;

    if ( mpParser == NULL )
    {
        mpParser = new DcgmFabricParser;
    }
    else
    {
        mpParser->fabricParserCleanup();
    }

    // try the topology file at the default location
    rc = mpParser->parseFabricTopology( DEFAULT_TOPOLOGY_FILE );
    if ( rc == FM_SUCCESS )
    {
        PRINT_INFO("%s", "Default topology file %s is parsed.", DEFAULT_TOPOLOGY_FILE);
        PRINT_INFO("%d %d %d %d %d %d",
                   "Node %d, Switch %d, Port %d, ReqEntry %d, RespEntry %d, GPU %d",
                   (int)mpParser->NodeCfg.size(), (int)mpParser->lwswitchCfg.size(),
                   (int)mpParser->portInfo.size(), (int)mpParser->reqEntry.size(),
                   (int)mpParser->respEntry.size(), (int)mpParser->gpuCfg.size());
        return rc;
    }

    mpParser->fabricParserCleanup();

    // try again with the deprecated topology file
    rc = mpParser->parseFabricTopology( DEPRECATED_TOPOLOGY_FILE );
    if ( rc == FM_SUCCESS )
    {
        PRINT_INFO("%s", "Deprecated topology file %s is parsed.", DEPRECATED_TOPOLOGY_FILE);
        PRINT_INFO("%d %d %d %d %d %d",
                   "Node %d, Switch %d, Port %d, ReqEntry %d, RespEntry %d, GPU %d",
                   (int)mpParser->NodeCfg.size(), (int)mpParser->lwswitchCfg.size(),
                   (int)mpParser->portInfo.size(), (int)mpParser->reqEntry.size(),
                   (int)mpParser->respEntry.size(), (int)mpParser->gpuCfg.size());
    }
    else
    {
        PRINT_ERROR("%d", "Failed to parse topology file rc %d.", rc);
    }

    return rc;
}

void
DcgmGlobalFabricManager::createFabricNodes(char *domainSocketPath)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    DcgmFabricNode *pFabricNode;

    mvFabricNodes.clear();

    for ( it = mpParser->NodeCfg.begin();
          it != mpParser->NodeCfg.end(); it++ )
    {
        pNode = it->second;
        if (domainSocketPath == NULL) {
            pFabricNode =  new DcgmFabricNode( pNode->IPAddress->c_str(),
                                               pNode->nodeId, this, false);
        } else {
            pFabricNode =  new DcgmFabricNode( domainSocketPath,
                                               pNode->nodeId, this, true);
        }
        mvFabricNodes.insert( std::make_pair(pNode->nodeId, pFabricNode) );
    }
}

bool
DcgmGlobalFabricManager::waitForAllFabricNodeCtrlConns(void)
{
    std::map <uint32_t, DcgmFabricNode *>::iterator it;
    std::map <uint32_t, DcgmFabricNode*> tempFabricNodes;

    // make a copy of the nods
    tempFabricNodes = mvFabricNodes;

    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
    const unsigned int WAIT_MS = 50; // wait interval
    unsigned int timeoutMs = 10000; // total time wait for the connection to establish

    // iterate over all the available nodes for connection
    while (true) {
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end();) {
            DcgmFabricNode *pFabricNode = it->second;
            if (pFabricNode->isControlConnectionActive()) {
                tempFabricNodes.erase(it++);
            } else {
                ++it;
            }
         }

        // one iteration is passed. exit if don't have any more nodes pending
        if (tempFabricNodes.empty()) {
            break;
        } else {
            timeNow = timelib_usecSince1970();
            if ((timeNow - timeStart) + WAIT_MS*1000 > timeoutMs*1000) {
                // elapsed all the time and still there are unconnected nodes.
                break;
            }
        }
        // wait for some time and poll nodes for connection again
        usleep(WAIT_MS * 1000);
    }

    // the above loop will finish when all the nodes are connected or due to a timeout
    // if there is a timeout, there will be unconnected nodes.
    if (tempFabricNodes.size()) {
        // there some unconnected nodes
        return false;
    }

    // all the nodes connected
    return true;
}

void
DcgmGlobalFabricManager::configureAllTrunkPorts(void)
{
    std::set<enum PortType> portTypes;

    portTypes.insert( TRUNK_PORT_SWITCH );
    mpConfig->configSwitchPortsWithTypes( portTypes );
}

void
DcgmGlobalFabricManager::sendGlobalConfigToAllFabricNodes(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;

    for ( it = mpParser->NodeCfg.begin();
        it != mpParser->NodeCfg.end(); it++ )
    {
        pNode = it->second;
        // first set nodeId and other information for the node
        mpConfig->sendNodeGlobalConfig( pNode->nodeId );
        // send all the peer nodes information (IP,NodeID etc) so that
        // peer LFM connection is established before link training.
        mpConfig->sendPeerLFMInfo( pNode->nodeId );
    }

    // Note: on multi-node system, wait for all the peer LFM connections to 
    // establish by checking for response messages
}

void
DcgmGlobalFabricManager::configureAllFabricNodes(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    std::set<enum PortType> portTypes;

     for ( it = mpParser->NodeCfg.begin();
           it != mpParser->NodeCfg.end(); it++ )
     {
        pNode = it->second;

        // do not send config to nodes that has no lwswitch
        if ( mLwswitchInfoMap[pNode->nodeId].size() == 0 )
        {
            PRINT_INFO("", "There is no lwswitch, no need to send configuration.");
            continue;
        }

        // configure access ports only here,
        // as trunk ports are already configured before link training
        portTypes.insert( ACCESS_PORT_GPU );
        FM_ERROR_CODE rc = mpConfig->configOneNode(pNode->nodeId, portTypes);
        if ( rc != FM_SUCCESS )
        {
            // set node config error state
            setNodeConfigError(pNode->nodeId);
        }
     }
}

FM_ERROR_CODE
DcgmGlobalFabricManager::waitForAllFabricNodeConfigCompletion(void)
{
    // wait for all nodes finish processing configuration
    std::map <uint32_t, DcgmFabricNode*>::iterator it;
    std::map <uint32_t, DcgmFabricNode*> tempFabricNodes;
    DcgmFabricNode *pFabricNode;

    // make a copy of the nodes
    tempFabricNodes = mvFabricNodes;

    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
    const unsigned int WAIT_MS = 50; // wait interval
    unsigned int timeoutMs = 10000; // total time wait for the configuration to establish

    // iterate over all the available nodes
    while (true) {
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end();) {
            pFabricNode = it->second;

            // check if there is any config error has happened
            if ( isNodeConfigErrorOclwred(pFabricNode->getNodeId()) == true )
            {
                PRINT_ERROR("%d","Config error on Node %d", pFabricNode->getNodeId());
                return FM_CFG_ERROR;
            }

            if (mpConfig->isPendingConfigReqEmpty(pFabricNode->getNodeId())) {
                tempFabricNodes.erase(it++);
            } else {
                mpConfig->dumpPendingConfigReq(pFabricNode->getNodeId());
                ++it;
            }
         }

        // one iteration is passed. exit if don't have any more nodes pending
        if (tempFabricNodes.empty()) {
            break;
        } else {
            timeNow = timelib_usecSince1970();
            if ((timeNow - timeStart) + WAIT_MS*1000 > timeoutMs*1000) {
                // elapsed all the time and still there are unfinished nodes.
                break;
            }
        }
        // wait for some time
        usleep(WAIT_MS * 1000);
    }

    if (tempFabricNodes.size()) {
        // some nodes have not finished configuration
        PRINT_ERROR("some nodes have not finished configuration","some nodes have not finished configuration");
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end(); it++) {
            pFabricNode = it->second;

            lwswitch::fmMessage errMsg;
            mpConfig->handleConfigError( pFabricNode->getNodeId(),
                                         ERROR_SOURCE_SW_GLOBALFM,
                                         ERROR_TYPE_CONFIG_TIMEOUT,
                                         errMsg );
            return FM_CFG_TIMEOUT;
        }
    }

    return FM_SUCCESS;
}

void
DcgmGlobalFabricManager::sendInitDoneToAllFabricNodes(void)
{
    // wait for all nodes finish processing configuration
    std::map <uint32_t, DcgmFabricNode*>::iterator it;
    DcgmFabricNode *pFabricNode;

    // send init done to all nodes, because all nodes have finished configuration
    for (it = mvFabricNodes.begin(); it != mvFabricNodes.end(); it++) {
        pFabricNode = it->second;
        mpConfig->sendConfigInitDoneReqMsg(pFabricNode->getNodeId());
    }
}

void
DcgmGlobalFabricManager::sendDeInitToAllFabricNodes(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;

     for ( it = mpParser->NodeCfg.begin();
           it != mpParser->NodeCfg.end(); it++ )
     {
        pNode = it->second;
        mpConfig->sendConfigDeInitReqMsg(pNode->nodeId);
     }
}

void
DcgmGlobalFabricManager::dumpAllGpuInfo(void)
{
    DcgmFMGpuInfoMap::iterator it;
    std::stringstream outStr;

    for ( it = mGpuInfoMap.begin(); it != mGpuInfoMap.end(); it++ ) {
        outStr << "Dumping GPU information for Node Index " << int(it->first) << std::endl;
        DcgmFMGpuInfoList gpuList = it->second;
        DcgmFMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            DcgmFMGpuInfo gpuInfo = (*jit);
            outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t Uuid: " << gpuInfo.uuid << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)gpuInfo.pciInfo.function << std::endl;
        }
    }

    for ( it = mBlacklistGpuInfoMap.begin(); it != mBlacklistGpuInfoMap.end(); it++ ) {
        outStr << "Dumping blacklisted GPU information for Node Index " << int(it->first) << std::endl;
        DcgmFMGpuInfoList gpuList = it->second;
        DcgmFMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            DcgmFMGpuInfo gpuInfo = (*jit);
            outStr << "\t gpuIndex: " << int(gpuInfo.gpuIndex) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)gpuInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)gpuInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)gpuInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)gpuInfo.pciInfo.function << std::endl;
            outStr << "\t\t uuid:" << std::hex << gpuInfo.uuid << std::endl;
            // log blacklisted gpu uuid information to syslog
            FM_SYSLOG_NOTICE("GPU uuid: %s is blacklisted", gpuInfo.uuid);
        }
    }

    std::string strInfo = outStr.str();
    if (strInfo.size() == 0) {
        strInfo = "No GPU Information is available\n";
    }

    PRINT_INFO("%s", "%s", strInfo.c_str());
}

void
DcgmGlobalFabricManager::dumpAllLWSwitchInfo(void)
{
    DcgmFMLWSwitchInfoMap::iterator it;
    DcgmFMLWSwitchInfoMap switchInfoMap = mLwswitchInfoMap;
    std::stringstream outStr;

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        outStr << "Dumping LWSwitch information for Node Index " << int(it->first) << std::endl;
        DcgmFMLWSwitchInfoList switchList = it->second;
        DcgmFMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            DcgmFMLWSwitchInfo switchInfo = (*jit);
            outStr << "\t switchIndex: " << int(switchInfo.switchIndex);
            outStr << "\t physicalId: " << std::hex << int(switchInfo.physicalId) << std::endl;
            outStr << "\t enabledLinkMask: " << std::hex << int(switchInfo.enabledLinkMask) << std::endl;
            outStr << "\t PCI Info:" << std::endl;
            outStr << "\t\t Domain:" << std::hex << (int)switchInfo.pciInfo.domain << std::endl;
            outStr << "\t\t Bus:" << std::hex << (int)switchInfo.pciInfo.bus << std::endl;
            outStr << "\t\t Device:" << std::hex << (int)switchInfo.pciInfo.device << std::endl;
            outStr << "\t\t Function:" << std::hex << (int)switchInfo.pciInfo.function << std::endl;
        }
    }

    std::string strInfo = outStr.str();
    if (strInfo.size() == 0) {
        strInfo = "No LWSwitch Information is available\n";
    }

    PRINT_INFO("%s", "%s", strInfo.c_str());
}

void
DcgmGlobalFabricManager::dumpLWLinkDeviceAndInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    for ( it = mpParser->NodeCfg.begin(); it != mpParser->NodeCfg.end(); it++ ) {
        pNode = it->second;
        PRINT_INFO("%d", "\nDumping LWLink information for Node Index %d\n", pNode->nodeId);
        DcgmFMLWLinkDevInfoList devList;
        mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        DcgmFMLWLinkDevInfoList::iterator jit;
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            DcgmFMLWLinkDevInfo devInfo = (*jit);
            std::stringstream outStr;
            devInfo.dumpInfo( &outStr );
            PRINT_INFO("%s", "%s", outStr.str().c_str());
        }
    }
}

void
DcgmGlobalFabricManager::dumpGpuConnMatrix(void)
{
    DcgmFMGpuLWLinkConnMatrixMap::iterator it;
    std::stringstream outStr;

    outStr << "Gpu LWLink Connection Matrix Map size = " << mGpuLWLinkConnMatrix.size() << std::endl;

    for (it = mGpuLWLinkConnMatrix.begin(); it != mGpuLWLinkConnMatrix.end(); it++) {
        DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
        DcgmFMGpuLWLinkConnMatrixList::iterator jit;
        for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
            DcgmFMGpuLWLinkConnMatrix gpuConnInfo = *jit;
            outStr << "GPU Physical Id = " << gpuConnInfo.gpuPhyIndex;
            outStr << " GPU Enum Index = " << gpuConnInfo.gpuEnumIndex;
            for ( int idx = 0; idx < DCGM_LWLINK_MAX_LINKS_PER_GPU; idx++ ) {
                outStr << " LnkSt[" << idx <<"]= " << gpuConnInfo.linkConnStatus[idx];
            }
            outStr << std::endl;
        }
    }

    PRINT_INFO("%s", "%s", outStr.str().c_str());
}

/*
 *  Get errors with errorMask from switches on nodeId
 *  errors will be allocated and queued to pErrQ, the caller should free the errors
 */
void
DcgmGlobalFabricManager::getNodeErrors( uint32_t nodeId, uint32_t errorMask )
{
    dcgmReturn_t rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::fmMessage *pResponse = NULL;
    lwswitch::targetSwitch *pTarget;
    lwswitch::getSwitchErrorRequest *pGetError = NULL;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;
    lwswitch::switchInfo *pCfg;
    int i;

    std::map <uint32_t, DcgmFabricNode *>::iterator nodeIt;
    nodeIt = mvFabricNodes.find( nodeId );
    if ( nodeIt == mvFabricNodes.end() )
    {
        PRINT_ERROR("%d","Invalid nodeId %d", nodeId);
        return;
    }

    PRINT_DEBUG("%d", "nodeId %d", nodeId);

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_GET_ERROR_REQ );

    pGetError = new lwswitch::getSwitchErrorRequest;
    pMessage->set_allocated_geterrorrequest( pGetError );

    for ( i = 0, it = mpParser->lwswitchCfg.begin();
          it != mpParser->lwswitchCfg.end();
          it++, i++ )
    {
        pCfg = it->second;
        if ( !pCfg || !pCfg->has_switchphysicalid() )
        {
            PRINT_ERROR("%d %d","Invalid switch cfg on nodeId %d, index %d",
                        nodeId, i);
            continue;
        }

        pTarget = pGetError->add_targetswitches();
        pTarget->set_targetswitchphysicalid( pCfg->switchphysicalid() );
    }

    DcgmFabricNode *pFabricNode = nodeIt->second;
    rc = pFabricNode->SendControlMessageSync( pMessage, &pResponse );
    if (  rc == DCGM_ST_OK )
    {
        if ( pResponse ) mControlMsgHndl->dumpMessage( pResponse );
        // parse the Error response
        // TODO - Calling parseErrorReportMsg will log the stats to DCGM cache manager.
        // Since this is an on-demand request to fetch stats, we need to decide on the 
        // use case and see whether the caller will provide the buffer to copy the stats.

        //mGlobalStatsMsgHndl->parseErrorReportMsg( pResponse, &mvFatalErrors, &mvNonFatalErrors );
    }
    else
    {
        PRINT_ERROR("%d, %d",
                    "Failed to send to nodeId %d, rc %d.",
                    nodeId, rc);
    }

    delete pMessage;
    if ( pResponse ) delete pResponse;
}

/*
 * Get stats from switches on nodeId, and save the stats in mvFabricStats
 */
void
DcgmGlobalFabricManager::getNodeStats( uint32_t nodeId )
{
    dcgmReturn_t rc;
    lwswitch::fmMessage *pMessage = NULL;
    lwswitch::fmMessage *pResponse = NULL;
    lwswitch::targetSwitch *pTarget;
    lwswitch::getSwitchLatencyHist *pGetLatency;
    lwswitch::getSwitchLwlinkCounter *pGetCounter;
    lwswitch::getNodeStatsRequest *pRequest = NULL;
    map <SwitchKeyType, lwswitch::switchInfo *>::iterator it;
    lwswitch::switchInfo *pCfg;
    int i;

    std::map <uint32_t, DcgmFabricNode *>::iterator nodeIt;
    nodeIt = mvFabricNodes.find( nodeId );
    if ( nodeIt == mvFabricNodes.end() )
    {
        PRINT_ERROR("%d","Invalid nodeId %d", nodeId);
        return;
    }

    PRINT_DEBUG("%d", "nodeId %d", nodeId);

    pMessage = new lwswitch::fmMessage();
    pMessage->set_type( lwswitch::FM_GET_NODE_STATS_REQ );

    pRequest = new lwswitch::getNodeStatsRequest;
    pMessage->set_allocated_getstatsrequest( pRequest );

    pGetLatency = pRequest->add_latestlatency();
    pGetCounter= pRequest->add_lwlinkcounter();

    for ( i = 0, it = mpParser->lwswitchCfg.begin();
          it != mpParser->lwswitchCfg.end();
          it++, i++ )
    {
        pCfg = it->second;
        if ( !pCfg || !pCfg->has_switchphysicalid() )
        {
            PRINT_ERROR("%d %d","Invalid switch cfg on nodeId %d, index %d",
                        nodeId, i);
            continue;
        }

        pTarget = pGetLatency->add_targetswitches();
        pTarget->set_targetswitchphysicalid( pCfg->switchphysicalid() );

        pTarget = pGetCounter->add_targetswitches();
        pTarget->set_targetswitchphysicalid( pCfg->switchphysicalid() );
    }

    DcgmFabricNode *pFabricNode = nodeIt->second;
    rc = pFabricNode->SendControlMessageSync( pMessage, &pResponse );
    if (  rc == DCGM_ST_OK )
    {
        if ( pResponse ) mControlMsgHndl->dumpMessage( pResponse );
        // parse the stats response
        // TODO - Calling parseStatsReportMsg will log the stats to DCGM cache manager.
        // Since this is an on-demand request to fetch stats, we need to decide on the 
        // use case and see whether the caller will provide the buffer to copy the stats.

        //mGlobalStatsMsgHndl->parseStatsReportMsg( pResponse, &mvFabricStats );
    }
    else
    {
        PRINT_ERROR("%d, %d",
                    "Failed to send to nodeId %d, rc %d.",
                    nodeId, rc);
    }

    delete pMessage;
    if ( pResponse ) delete pResponse;
}

void
DcgmGlobalFabricManager::queueErrorWorkerRequest(uint32 nodeId, lwswitch::fmMessage *errorMsg)
{
    // As part of error handling, we may send a re-train/FM session clear request
    // to corresponding LFM and wait for it to complete. But we are lwrrently in the context
    // of LibEvent thread ie LwcmClientListener::ReadCB() which triggered the process message.
    // Since Libevent is initialized as thread safe, when the ReadCB is ilwoked Libevent is
    // holding a lock. So in a single-node system, if the GFM try to send a message
    // (ie a train_request), it won't process/delivered to LFM by Libevent, causing an indefinite wait.

    // Make the error processing to use a worker thread and return from the LibEvent context.
    // Later the worker thread will process the actual error handling.
    job_t *pJob = new job_t;
    FmErrorWorkerReqInfo *pWorkerReqInfo = new FmErrorWorkerReqInfo;
    pWorkerReqInfo->pGfmObj = this;
    pWorkerReqInfo->errorMsg = *errorMsg;
    pWorkerReqInfo->nodeId = nodeId;

    pJob->job_function = DcgmGlobalFabricManager::processErrorWorkerRequest;
    pJob->user_data = pWorkerReqInfo;
    workqueue_add_job(&mErrWorkQueue, pJob);
}

void
DcgmGlobalFabricManager::processErrorWorkerRequest(job_t *pJob)
{
    FmErrorWorkerReqInfo *pWorkerReqInfo;
    pWorkerReqInfo = (FmErrorWorkerReqInfo*) pJob->user_data;
    DcgmGlobalFabricManager *pGfmObj = pWorkerReqInfo->pGfmObj;
    lwswitch::fmMessage &errorMsg = pWorkerReqInfo->errorMsg;
    uint32 nodeId = pWorkerReqInfo->nodeId;
    GlobalFMErrorTypes errType = ERROR_TYPE_MAX;
    GlobalFMErrorSource errSource = ERROR_SOURCE_MAX;

    // process each error based on the error type reported
    switch(errorMsg.type()) {
        case lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY: {
            errSource = ERROR_SOURCE_LWSWITCH;
            errType = ERROR_TYPE_LWLINK_RECOVERY;
            break;
        }
        case lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY: {
            errSource = ERROR_SOURCE_GPU;
            errType = ERROR_TYPE_LWLINK_RECOVERY;
            break;
        }
        case lwswitch::FM_LWLINK_ERROR_GPU_FATAL: {
            errSource = ERROR_SOURCE_GPU;
            errType = ERROR_TYPE_LWLINK_FATAL;
            break;
        }
        case lwswitch::FM_LWSWITCH_ERROR_FATAL: {
            errSource = ERROR_SOURCE_LWSWITCH;
            errType = ERROR_TYPE_FATAL;
            break;
        }
        default: {
            PRINT_ERROR("", "GlobalFM: unknown error type information in error worker request");
            delete pWorkerReqInfo;
            delete pJob;
            return;
        }
    }

    // ilwoke the actual error handler
    DcgmGlobalFMErrorHndlr errHndlr(pGfmObj, nodeId, 0, errSource, errType, errorMsg);
    errHndlr.processErrorMsg();

    delete pWorkerReqInfo;
    delete pJob;
}

bool
DcgmGlobalFabricManager::getGpuPhysicalIndex(uint32 nodeId, uint32_t enumIndex, uint32_t &physicalIdx)
{
    DcgmFMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
    DcgmFMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        DcgmFMGpuLWLinkConnMatrix gpuConnInfo = *jit;
        if (gpuConnInfo.gpuEnumIndex == enumIndex) {
            physicalIdx = gpuConnInfo.gpuPhyIndex;
            return true;
        }
    }

    return false;
}

bool
DcgmGlobalFabricManager::getGpuEnumIndex(uint32 nodeId, uint32_t physicalIdx, uint32_t &enumIndex)
{
    DcgmFMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
    DcgmFMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        DcgmFMGpuLWLinkConnMatrix gpuConnInfo = *jit;
        if (gpuConnInfo.gpuPhyIndex == physicalIdx) {
            enumIndex = gpuConnInfo.gpuEnumIndex;
            return true;
        }
    }

    return false;
}

bool
DcgmGlobalFabricManager::getGpuPciBdf(uint32 nodeId, uint32_t enumIndex, DcgmFMPciInfo &pciInfo)
{
    DcgmFMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    DcgmFMGpuInfoList::iterator it;
    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        DcgmFMGpuInfo tempInfo = (*it);
        if (tempInfo.gpuIndex == enumIndex) {
            pciInfo = tempInfo.pciInfo;
            return true;
        }
    }

    // not found the pci bdf
    return false;
}

bool
DcgmGlobalFabricManager::getGpuUuid(uint32 nodeId, uint32 physicalId, char uuid[])
{
    uint32_t gpuEnumIndex;
    uint32_t isFound = false;

    // first get enumeration index of the gpu from our physicalId to enumIdx mapping
    DcgmFMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    DcgmFMGpuLWLinkConnMatrixList connMatrixList = it->second;
    DcgmFMGpuLWLinkConnMatrixList::iterator jit;
    for (jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++) {
        DcgmFMGpuLWLinkConnMatrix gpuConnInfo = (*jit);
        if (gpuConnInfo.gpuPhyIndex == physicalId) {
            isFound = true;
            gpuEnumIndex = gpuConnInfo.gpuEnumIndex;
            break;
        }
    }

    // return if not able to locate the physical id
    if (!isFound) {
        return false;
    }

    // now copy the uuid information from GPU information
    DcgmFMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    DcgmFMGpuInfoList::iterator git;
    for (git = gpuList.begin(); git != gpuList.end(); git++) {
        DcgmFMGpuInfo tempInfo = (*git);
        if (tempInfo.gpuIndex == gpuEnumIndex) {
            strncpy(uuid, tempInfo.uuid, sizeof(tempInfo.uuid));
            return true;
        }
    }

    // not found the uuid
    return false;
}

/*************************************************************************************
 * For a given LWSwitch Physical ID, find the corresponding device's ID in the
 * LWLinkCoreLib driver. LWSwitch driver and CoreLib driver uses different IDs and 
 * the common factor between them is PCI BDF. CoreLib driver's device IDs are derived
 * from PCI BDF.
**************************************************************************************/
bool
DcgmGlobalFabricManager::getLWSwitchLWLinkDriverId(uint32 nodeId, uint32 physicalId, uint64 &lwLinkSwitchId)
{
    DcgmFMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);

    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    // find the corresponding LWSwitch's PCI BDF
    DcgmFMLWSwitchInfoList switchList = it->second;
    DcgmFMLWSwitchInfoList::iterator jit;
    DcgmFMLWSwitchInfo switchInfo;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        switchInfo = (*jit);
        if (switchInfo.physicalId == physicalId ) {
            // found the desired LWSwitch
            break;
        }
    }

    if (jit == switchList.end()) {
        // not found the specified LWSwitch physical id
        return false;
    }

    // we now have the required LWSwitch PCI BDF, using that
    // do the lookup on devices returned LWLinkCoreLib driver.
    DcgmFMLWLinkDevInfo lwLinkDevInfo;
    if (mLWLinkDevRepo.getDeviceInfo(nodeId, switchInfo.pciInfo, lwLinkDevInfo)) {
        lwLinkSwitchId = lwLinkDevInfo.getDeviceId();
        return true;
    }

    return false;
}

void
DcgmGlobalFabricManager::addSwitchesToCacheManager(void)
{
    DcgmFMLWSwitchInfoMap::iterator it;
    DcgmFMLWSwitchInfoMap switchInfoMap = mLwswitchInfoMap;
    
    //Add local LwSwitches to the cache manager
    if( mLwswitchInfoMap.size() > 1) {
        //TODO cache manager/DCGM interactions not yet ready for multi-node
        PRINT_DEBUG("%ld", "%ld node system, no DCGM interaction for now.", mLwswitchInfoMap.size());
        return;
    }

    //TODO local node ID may not be zero...how do we find local node?
    it = switchInfoMap.find(0);
    if(it == switchInfoMap.end())
    {
        PRINT_DEBUG("", "No switches present for local node.");
        return;
    }
    
    DcgmFMLWSwitchInfoList switchList = it->second;
    DcgmFMLWSwitchInfoList::iterator jit;

    for (jit = switchList.begin(); jit != switchList.end(); jit++) 
    {
        DcgmFMLWSwitchInfo switchInfo = (*jit);
        mpCacheManager->AddLwSwitch(switchInfo.physicalId);
    }
}

void
DcgmGlobalFabricManager::pruneNonDetectedLWSwitches(void)
{
    int disabledSwitchCount = mTopoValidator->disableSwitches();
    if ( disabledSwitchCount > 0 ) {
        PRINT_INFO("%d", "%d number of LWSwitches are disabled.", disabledSwitchCount);
    }

#ifdef DEBUG
    // gather a list of disabled switches or GPUs from a conf file (manual disable)
    mpParser->parseFabricTopologyConf(DEFAULT_TOPOLOGY_CONF_FILE);
#endif

    // modify the fabric if there are disabled switches, before configuring trunk ports
    if ( mpParser->getNumDisabledSwitches() > 0 ) {
        mpParser->modifyFabric(DcgmFabricParser::PRUNE_SWITCH);
    }
}

void
DcgmGlobalFabricManager::pruneNonDetectedGpus(void)
{
    // gather a list of disabled GPUs that are not validated
    int disabledGpuCount = mTopoValidator->disableGpus(mGpuLWLinkConnMatrix);
    if ( disabledGpuCount > 0 ) {
        PRINT_INFO("%d", "%d number of GPUs are disabled.", disabledGpuCount);
    }

    // modify the fabric if there are disabled switches or GPUs, before configuring nodes
    if ( mpParser->getNumDisabledGpus() > 0 ) {
        mpParser->modifyFabric(DcgmFabricParser::PRUNE_GPU);
    }
}

void
DcgmGlobalFabricManager::disableLWSwitchTrunkLinks(void)
{
    // For DGX-2 and HGX-2, trunk links are 0,1,2,3,8,9,10,11 for all the LWSwitches.
    // Note: Change this mask based on platform.
    uint64 disableMask = 0xF0F;

    if (!mpParser->isSwtichGpioPresent()) {
        // skip if switch GPIO IDs are not specified in topology file
        return;
    }

    if( mLwswitchInfoMap.size() > 1) {
        //Do not disable trunk links for Multi-node
        PRINT_DEBUG("%ld", "%ld node system, ignore disabling trunk LWLinks", mLwswitchInfoMap.size());
        return;
    }

    // Disable links only for HGX-2 Multi-host systems (ie only 6 LWSwitches)
    if (mLwswitchInfoMap[0].size() != 6) {
        return;
    }

    //
    // Skip if the trunk links are already disabled in LWSwitch to prevent IOCTL errors.
    // This can happen when FM is restarted (i.e. FM initially removed the links from LWSwitch driver).
    // Until the driver is re-loaded or the LWSwitches are re-enumerated, driver will not have
    // trunk links and any attempt to remove them will result in IOCTL failures.
    //

    // TODO fix this for multi-node case and also where node ID 0 is not used
    DcgmFMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(0);
    DcgmFMLWSwitchInfoList &switchList = it->second;
    DcgmFMLWSwitchInfo switchInfo = switchList.front();
    if (!(switchInfo.enabledLinkMask & disableMask)) {
        // Trunk links are not present in driver, no need to disable them.
        return;
    }

    PRINT_INFO("", "Disabling LWSwitch trunk links.");

    // go ahead and disable trunk links
    FM_ERROR_CODE rc;
    uint32_t nodeId = 0; //single node for now
    rc = mpConfig->configDisableSwitchLinks(nodeId, disableMask);
    if ( rc != FM_SUCCESS ) {
        PRINT_ERROR("", "Failed to disable LWSwitch trunk links to support multi-host configuration\n");
        throw std::runtime_error("Failed to disable LWSwitch trunk links to support multi-host configuration");
    }

    // Update GFM LWSwitch link mask information accordingly. (or we should fetch it again from LocalFM)
    // Note: switchList is accessed as a reference.
    DcgmFMLWSwitchInfoList::iterator jit;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        DcgmFMLWSwitchInfo &switchInfo = (*jit);
        switchInfo.enabledLinkMask = (switchInfo.enabledLinkMask & ~disableMask);
    }
}

void
DcgmGlobalFabricManager::initializeAllLWLinksAndTrainConns(void)
{
    // Do LWLink initialization for all nodes
    DcgmGFMHelper::lwLinkInitializeAllNodes(mLinkTrainIntf, mpParser, mLWLinkDevRepo);
    dumpLWLinkDeviceAndInitStatus();

    // discover all the intra node connections
    DcgmGFMHelper::lwLinkDiscoverIntraNodeConnOnNodes(mpParser, mLinkTrainIntf, mLWLinkConnRepo);
    // discover all the inter node connections
    DcgmGFMHelper::lwLinkDiscoverInterNodeConnections(mLinkTrainIntf, mpParser, mLWLinkConnRepo);

    // populate all inter node connections on the respective nodes
    DcgmGFMHelper::lwlinkAddInterNodeConnections(mLinkTrainIntf, mLWLinkConnRepo, mLWLinkDevRepo);

    if (isRestart())
    {
        // do not train links at restart
        return;
    }

    // train all connections to high speed
    DcgmGFMHelper::lwLinkTrainIntraNodeConnections(mLinkTrainIntf, mLWLinkConnRepo,
                                                   mLWLinkDevRepo, LWLINK_TRAIN_SAFE_TO_HIGH);

    DcgmGFMHelper::lwLinkTrainInterNodeConnections(mLinkTrainIntf, mLWLinkConnRepo,
                                                   mLWLinkDevRepo, LWLINK_TRAIN_SAFE_TO_HIGH);

    mLWLinkConnRepo.dumpAllConnAndStateInfo(mLWLinkDevRepo);
}

void
DcgmGlobalFabricManager::startNonSharedFabricMode(void)
{
    // Non-Shared Fabric Mode support the following configurations
    // 1. Bare Metal
    // 2. HGX-2 Multi-host
    // 3. Full pass-through (GPU & LWSwitch) multi-tenancy

    // prune switches that are in the topology but not discovered by the node
    // this can happen in Multi-host/multi-tenancy mode
    pruneNonDetectedLWSwitches();

    // In Multi-host, we should disable all the trunk links from switch driver
    // to prevent driver from initializing them as part of link training.
    disableLWSwitchTrunkLinks();

    // Query all the device/links from LWLinkCoreLib driver after disabling unwanted links.
    DcgmGFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);

    // configure all switch trunk ports before link training start
    // access ports would be pruned, and will be configured later,
    // after GPU and access port pruning is determined from discovered topology
    configureAllTrunkPorts();

    addSwitchesToCacheManager();

    // initialize all the links, discover connections and train them to high speed.
    initializeAllLWLinksAndTrainConns();

    // build GPU link status mapping based on detected LWLink connections.
    // the corresponding GPU link status will be false if the connection is not in Active state.
    // the GPU will be later pruned by the below disableGpus()'s logic.
    // Do this mapping first to pouplate corresponding GPU's link enabled mask.
    mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
    dumpGpuConnMatrix();

    // log any LWLink initialization failures
    mTopoValidator->checkLWLinkInitStatusForAllDevices();

    // do all the validation after link initialization and training 
    // so that all the failure is reported

    // validate the topology information
    if ( !mTopoValidator->validateTopologyInfo() ) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        //Multi-node topologies need to be more flexible handling failures hence simple topology
        //validation may not work. Need to re-examine this code when we have further designed for multi-node
        if(mLwswitchInfoMap.size() < 2)
#endif
        {
            PRINT_ERROR("", "Detected LWSwitch/LWLink trunk connection information does not match topology\n");
            throw std::runtime_error("Detected LWSwitch/LWLink trunk connection information does not match topology");
        }
    }
    PRINT_DEBUG("%lu", "Number of nodes=%lu", mLwswitchInfoMap.size());

    // validate LWLink trunk connection states
    if (!mTopoValidator->isAllLWLinkTrunkConnsActive()) {
        FM_SYSLOG_ERR("Not all the LWLink trunk (LWSwitch to LWSwitch) connections are in Active state\n");
        throw std::runtime_error("Not all the LWLink trunk (LWSwitch to LWSwitch) connections are in Active state");
    }

    // publish LWLink link state of all devices to dcgm cache manager
    publishLinkStatetoCacheManager();

    // remove all the non-detected GPUs from fabric config context to
    // prevent any config attempt.
    pruneNonDetectedGpus();

    // configure all switch ports and routing tables on all nodes
    // all routing entries are valid in non-shared fabric mode
    configureAllFabricNodes();
}

void
DcgmGlobalFabricManager::startSharedFabricMode(void)
{
    // prune switches that are in the topology but not discovered by the node
    // this can happen in Multi-host
    pruneNonDetectedLWSwitches();

    // In Multi-host, we should disable all the trunk links from switch driver
    // to prevent driver from initializing them as part of link training.
    disableLWSwitchTrunkLinks();

    // Query all the device/links from LWLinkCoreLib driver.
    DcgmGFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);

    // configure all switch trunk ports before link training start
    configureAllTrunkPorts();

    addSwitchesToCacheManager();

    // initialize all the links, discover connections and train them to high speed.
    initializeAllLWLinksAndTrainConns();

    // build GPU link status mapping based on detected LWLink connections.
    // the corresponding GPU link status will be false if the connection is not in Active state.
    // the GPU will be later pruned by the below disableGpus()'s logic.
    // Do this mapping first to pouplate corresponding GPU's link enabled mask.
    mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
    dumpGpuConnMatrix();

    // log any LWLink initialization failures
    mTopoValidator->checkLWLinkInitStatusForAllDevices();

    // do all the validation after link initialization and training 
    // so that all the failure is reported

    // validate the topology information
    if ( !mTopoValidator->validateTopologyInfo() ) {
        PRINT_ERROR("", "Detected LWSwitch/LWLink trunk connection information does not match topology\n");
        throw std::runtime_error("Detected LWSwitch/LWLink trunk connection information does not match topology");
    }

    // validate LWLink trunk connection states
    if (!mTopoValidator->isAllLWLinkTrunkConnsActive()) {
        FM_SYSLOG_ERR("Not all the LWLink trunk (LWSwitch to LWSwitch) connections are in Active state\n");
        throw std::runtime_error("Not all the LWLink trunk (LWSwitch to LWSwitch) connections are in Active state");
    }

    // Note: no need to publish LWLink link state of all devices to dcgm cache manager
    // as the links will be disabled after initialization.

    // remove all the non-detected GPUs from fabric config context to
    // prevent any config attempt.
    pruneNonDetectedGpus();

    // create fabric partition manager object
    mGfmPartitionMgr = new DcgmGFMFabricPartitionMgr(this);
    mGfmPartitionMgr->buildPartitionMappings();

    // configure all switch ports and routing tables on all nodes
    // all routing entries are invalid in shared fabric mode
    configureAllFabricNodes();

    // reset all LWSwitch links before detaching the GPUs. Before activating the partition,
    // GPUs will be SBRed by Hypervisor. So, both GPU and LWSwitch side LWLinks will be in INIT
    // state for training for partition activation. Also train the connections to OFF
    // before starting reset to avoid GPU side links from flagging errors.

    // Note: Not bailing out on error links power down. Later when a partition with corresponding
    // LWLinks is activated, that activation may/may not fail. Also, there may be other partitions
    // which are not using the failed LWLinks, which can be activated successfully.
    DcgmGFMHelper::lwLinkTrainIntraNodeConnections(mLinkTrainIntf, mLWLinkConnRepo,
                                                   mLWLinkDevRepo, LWLINK_TRAIN_TO_OFF);

    DcgmGFMHelper::lwLinkResetAllSwitchLinks(0, mLinkTrainIntf);

    mpConfig->configSharedLWSwitchPartitionDetachGPUs(0, ILWALID_FABRIC_PARTITION_ID);


}

void
DcgmGlobalFabricManager::restartSharedFabricMode(void)
{
    // Query all the device/links from LWLinkCoreLib driver after restart
    DcgmGFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);

    addSwitchesToCacheManager();

    // initialize all the links, discover connections, but do NOT train them.
    initializeAllLWLinksAndTrainConns();

    addSwitchesToCacheManager();
}

void
DcgmGlobalFabricManager::publishLinkStatetoCacheManager(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    for ( it = mpParser->NodeCfg.begin(); it != mpParser->NodeCfg.end(); it++ ) {
        pNode = it->second;
        bool retVal = mLWLinkDevRepo.publishNodeLinkStateToCacheManager(pNode->nodeId, this);
        if (!retVal) {
            PRINT_ERROR("%d", "failed to publish LWLink state to cache manager for node %d", pNode->nodeId);
            // continue with rest of the nodes
        }
    }
}

dcgmReturn_t
DcgmGlobalFabricManager::getSupportedFabricPartitions(dcgmFabricPartitionList_t &dcgmFabricPartitions)
{
    // these APIs are supported only in shared fabric mode.
    if (mSharedFabric == false) {
        return DCGM_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mFabricManagerInitDone == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if (mpHaMgr->isInitDone() == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    // all looks good, ilwoke the actual partition manager API.
    return mGfmPartitionMgr->getPartitions(dcgmFabricPartitions);
}

dcgmReturn_t
DcgmGlobalFabricManager::activateFabricPartition(unsigned int partitionId)
{
    // these APIs are supported only in shared fabric mode.
    if (mSharedFabric == false) {
        return DCGM_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mFabricManagerInitDone == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if (mpHaMgr->isInitDone() == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual partition manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    return mGfmPartitionMgr->activatePartition(0, partitionId);
}

dcgmReturn_t
DcgmGlobalFabricManager::deactivateFabricPartition(unsigned int partitionId)
{
    // these APIs are supported only in shared fabric mode.
    if (mSharedFabric == false) {
        return DCGM_ST_NOT_SUPPORTED;
    }

    // proceed only after we are done with all the FM initialization sequence
    if (mFabricManagerInitDone == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    // proceed only after HaMgr completes HA initialization
    if (mpHaMgr->isInitDone() == false) {
        return DCGM_ST_NOT_CONFIGURED;
    }

    //
    // all looks good, ilwoke the actual partition manager API.
    // nodeId is set to 0 before multi-node implementation is in place.
    //
    return mGfmPartitionMgr->deactivatePartition(0, partitionId);
}

dcgmReturn_t
DcgmGlobalFabricManager::setActivatedFabricPartitions(dcgmActivatedFabricPartitionList_t &dcgmFabricPartitions)
{
    // this API is supported only in shared fabric mode and when FM is restarted
    if ((mSharedFabric == false) || (mRestart == false)) {
        return DCGM_ST_NOT_SUPPORTED;
    }

    // this API should only be called before HA mgr init done
    if (mpHaMgr->isInitDone() == true) {
        return DCGM_ST_NOT_SUPPORTED;
    }

    // no need to wait for HaMgr completes HA initialization
    // because this API is part of the HA restart process

    return mGfmPartitionMgr->setActivatedPartitions(dcgmFabricPartitions);
}

uint32_t
DcgmGlobalFabricManager::getControlMessageRequestId(uint32_t fabricNodeId)
{
    std::map <uint32_t, DcgmFabricNode *>::iterator it;

    it = mvFabricNodes.find( fabricNodeId );
    if ( it == mvFabricNodes.end() )
    {
        PRINT_DEBUG("%d", "Invalid fabric node ID %d", fabricNodeId);
        return 0;
    }

    DcgmFabricNode *pFabricNode = it->second;
    return pFabricNode->getControlMessageRequestId();
}

void
DcgmGlobalFabricManager::setNodeConfigError(uint32_t nodeId)
{
    std::map <uint32_t, DcgmFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        DcgmFabricNode *pNode = it->second;
        pNode->setConfigError();
    }
}

void
DcgmGlobalFabricManager::clearNodeConfigError(uint32_t nodeId)
{
    std::map <uint32_t, DcgmFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        DcgmFabricNode *pNode = it->second;
        pNode->clearConfigError();
    }
}

bool
DcgmGlobalFabricManager::isNodeConfigErrorOclwred(uint32_t nodeId)
{
    std::map <uint32_t, DcgmFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        DcgmFabricNode *pNode = it->second;
        return pNode->isConfigError();
    }

    return false;
}



