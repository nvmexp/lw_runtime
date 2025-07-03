/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <sstream>
#include <iomanip>

#include "fm_config_options.h"
#include "topology.pb.h"
#include "FMErrorCodesInternal.h"
#include "GlobalFabricManager.h"
#include "FmRequest.h"
#include "GFMHelper.h"
#include "fm_log.h"
#include "GlobalFmErrorStatusMsgHndlr.h"
#include "FMTopologyValidator.h"
#include "GlobalFmCommandServer.h"
#include "GlobalFmErrorHndlr.h"
#include "GFMFabricPartitionMgr.h"
#include "GlobalFMLibCmdServer.h"
#include "GlobalFMInternalCmdServer.h"
#include "FMDeviceProperty.h"
#include <g_lwconfig.h>

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
#define NUM_LINK_TRAIN_RETRYS 5
#endif

GlobalFabricManager::GlobalFabricManager(GlobalFmArgs_t *pGfmArgs)
{
    //
    // create and initialize all the globalFM class objects. this initialization
    // can throw exception. catch the same and do required clean-up as
    // destructor will not be called in this.
    //

    // set all the member pointers which requires explicit clean-up to null.
    mGpuInfoMap.clear();
    mGpuLWLinkSpeedInfoMap.clear();
    mExcludedGpuInfoMap.clear();
    mLwswitchInfoMap.clear();
    mExcludedLwswitchInfoMap.clear();
    mMsgHandlers.clear();
    mvFabricNodes.clear();

    mGfmPartitionMgr = NULL;
    mpParser = NULL;
    mpHaMgr = NULL;
    mpDegradedModeMgr = NULL;
    mGlobalFmApiHandler = NULL;
    mGlobalFMLibCmdServer = NULL;
    mGlobalFMInternalCmdServer = NULL;
    mFmLibCmdBindInterface = NULL;
    mFmLibCmdSockPath = NULL;
    mTopoValidator = NULL;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    mFabricNodeConfigFile = NULL;
    mMultiNodeTopology = NULL;
#endif
    mFmLWlinkRetrainCount = 0;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    mpMcastMgr = NULL;
#endif

    mGlobalCmdServer = NULL;

    try {
        doGlobalFMInitialization(pGfmArgs);
    } catch (const std::runtime_error &e) {
        cleanup();

        // finally pass the exception to caller
        throw std::runtime_error(e.what());
    }
}

void GlobalFabricManager::doGlobalFMInitialization(GlobalFmArgs_t *pGfmArgs)
{
    FMIntReturn_t rc;

    mFabricManagerInitDone = false;

    mFabricMode = pGfmArgs->fabricMode;
    mFabricModeRestart = pGfmArgs->fabricModeRestart;
    mStartingPort = pGfmArgs->fmStartingTcpPort;
    mStagedInit = pGfmArgs->stagedInit;

    mFmLibCmdBindInterface = strdup(pGfmArgs->fmLibCmdBindInterface);
    mFmLibCmdSockPath = strdup(pGfmArgs->fmLibCmdSockPath);
    mFmLibPortNumber = pGfmArgs->fmLibPortNumber;
    mContinueWithFailure = pGfmArgs->continueWithFailures;
    mEnableTopologyValidation = pGfmArgs->enableTopologyValidation;
    mDisableDegradedMode = pGfmArgs->disableDegradedMode;
    mDisablePruning = pGfmArgs->disablePruning;
    mGfmWaitTimeout = pGfmArgs->gfmWaitTimeout;
	mSimMode = pGfmArgs->simMode;
    mFmLWlinkRetrainCount = (pGfmArgs->fmLWlinkRetrainCount < 0) ?
        0 : pGfmArgs->fmLWlinkRetrainCount;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    mDisableLwlinkAli = pGfmArgs->disableLwlinkAli;
#endif

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (pGfmArgs->fabricNodeConfigFile != NULL) {
        mFabricNodeConfigFile = strdup(pGfmArgs->fabricNodeConfigFile);
    }

    if (pGfmArgs->multiNodeTopology != NULL) {
        mMultiNodeTopology = strdup(pGfmArgs->multiNodeTopology);

        // Use default multinode retry count if nothing is specified in config
        mFmLWlinkRetrainCount = (pGfmArgs->fmLWlinkRetrainCount < 0) ?
            NUM_LINK_TRAIN_RETRYS : pGfmArgs->fmLWlinkRetrainCount;
    }
#endif
    
    //
    // If GFM exited (due to some error) before reaching to the connection dump, 
    // user may be looking at a stale file. Hence deleting the file. Check if file
    // exists before deleting.
    //
    ifstream dumpFile;
    dumpFile.open(MULTI_NODE_LWLINK_CONN_DUMP_FILE);
    if (dumpFile) {
        dumpFile.close();
        if (remove(MULTI_NODE_LWLINK_CONN_DUMP_FILE) != 0) {
            FM_LOG_WARNING("error deleting the trunk lwlink connection dump file %s", MULTI_NODE_LWLINK_CONN_DUMP_FILE);
        }
    }

    lwosInitializeCriticalSection( &mLock );

    // initialize error handling work queue with one worker thread
    if (workqueue_init(&mErrWorkQueue, 1)) {
        FM_LOG_ERROR("failed to create global fabric manager GPU and LWSwitch error processing worker queue");
        throw std::runtime_error("failed to create global fabric manager GPU and LWSwitch error processing worker queue");
    }

    if (((mFabricMode != FM_MODE_SHARED_LWSWITCH) && (mFabricMode != FM_MODE_VGPU)) && (mFabricModeRestart == true)) {
        FM_LOG_ERROR("fabric manager restart option is only supported for Shared LWSwitch or vGPU based multitenancy mode");
        throw std::runtime_error("fabric manager restart option is only supported for shared LWSwitch or vGPU based multitenancy mode");
    }

    // vGPU Mode - Fabric Manager config option with reduced bandwidth is not supported in vGPU multitenanc mode
    if ((mFabricMode == FM_MODE_VGPU) && ((pGfmArgs->accessLinkFailureMode == 0x1) || (pGfmArgs->trunkLinkFailureMode == 0x1)
        || (pGfmArgs->lwswitchFailureMode == 0x1))) {
        FM_LOG_ERROR("fabric manager config option with reduced bandwidth is not supported in vGPU multitenancy mode");
        FM_SYSLOG_ERR("fabric manager config option with reduced bandwidth is not supported in vGPU multitenancy mode");
        throw std::runtime_error("fabric manaher config option with reduced bandwidth is not supported in vGPU multitenancy mode");
    } 

    // create all the message handlers
    mpConfig = new FMFabricConfig( this );
    mMsgHandlers.push_back(mpConfig);

    mLinkTrainIntf = new GlobalFMLWLinkIntf(this);
    mMsgHandlers.push_back(mLinkTrainIntf);

    mDevInfoMsgHndlr = new GlobalFMDevInfoMsgHdlr(this);
    mMsgHandlers.push_back(mDevInfoMsgHndlr);

    mGlobalErrStatusMsgHndl = new GlobalFmErrorStatusMsgHndlr(this);
    mMsgHandlers.push_back(mGlobalErrStatusMsgHndl);

#ifdef DEBUG
#if !defined(LW_MODS) || defined(LW_MODS_GDM_BUILD)
    // start our localFM command server interface
    mGlobalCmdServer = new GlobalFMCommandServer(this);
#endif
#endif

    // create our topology validator instance
    mTopoValidator = new FMTopologyValidator(this);

    // Parse from the topology
    mpParser = new FMFabricParser(pGfmArgs->fabricPartitionFileName);


    //
    // Using the MULTI_NODE_TOPOLOGY option, we can decide if the user wants to configure the system
    // for single node or multi node. If the option is empty, then it means that the user wants FM to configure
    // a single node and in case of an appropriate value, ex. dgxa100_all_to_all_9node_topology, user
    // wants FM to configure multi node. The option also has the corresponding architecture type embedded into
    // it. So in the given example, we know that the architecture of the system is Ampere/LR. 
    // TODO: Architecture type can be filled for using the multiNodeTopologyType and can be verified by getting
    // switch info from all nodes further below
    //

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool isMultiNode = isMultiNodeMode();
    if (true == isMultiNode) {
        doMultiNodeGlobalFMPreInit(pGfmArgs);
    } 
    else
#endif
    doSingleNodeGlobalFMPreInit(pGfmArgs);

    // at this stage, a topology file (either single node or multi-node) is parsed

    // send all the general (non-switch/gpu) config to all nodes
    sendPeerLFMInfoToAllFabricNodes();

    //
    // gather all GPU and LWSwitch device info, including physicalIds from all nodes.
    // get LWLink device information from CoreLib driver after disabling any links (like trunk links in multi-host)
    //
    GFMHelper::getLWSwitchDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLwswitchInfoMap, mExcludedLwswitchInfoMap);
    GFMHelper::getGpuDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mGpuInfoMap, mExcludedGpuInfoMap);
 
    dumpAllGpuInfo();
    dumpAllLWSwitchInfo();

    uint32_t numberBaseboard = GFMHelper::getNumBaseboard(0, mLwswitchInfoMap, mExcludedLwswitchInfoMap);
    FM_LOG_INFO("number of GPU base board detected: %d", numberBaseboard);

    //
    // FM has nothing to do in Non-LWSwitch environment. So abort it.
    // Go through the map to see at least one node has an LWSwitch.
    //
    FMLWSwitchInfoMap::iterator it;
    for (it = mLwswitchInfoMap.begin(); it != mLwswitchInfoMap.end(); it++) {
        if(it->second.size() != 0) {
            // if any node has LWSwitches GFM should run to manage a multi-node system
            break;
        }
    }
    if(it == mLwswitchInfoMap.end()) {
        FM_LOG_ERROR("No LWSwitches detected, aborting fabric manager");
        FM_SYSLOG_ERR("No LWSwitches detected, aborting fabric manager");
        throw std::runtime_error("No LWSwitches detected, aborting fabric manager");
    }

    // create fabric partition manager
    if ((mFabricMode == FM_MODE_SHARED_LWSWITCH) || (mFabricMode == FM_MODE_VGPU)) {
        mGfmPartitionMgr = new GlobalFMFabricPartitionMgr(this);
    }

    // create multicast manager
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    mpMcastMgr = new GlobalFmMulticastMgr(this);
#endif

    // Create HA manager
    mpHaMgr = new GlobalFmHaMgr(this, pGfmArgs->stateFileName);

    if ( ( isFabricModeRestart() == true ) &&
         ( mpHaMgr->validateAndLoadState() == false ) ) {
        std::ostringstream ss;
        ss << "failed to validate and load fabric manager states during restart mode";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // Create degraded mode manager
    mpDegradedModeMgr = new GlobalFmDegradedModeMgr(this, pGfmArgs->accessLinkFailureMode,
                                                    pGfmArgs->trunkLinkFailureMode,
                                                    pGfmArgs->lwswitchFailureMode);

    // done with all the common initialization sequence. branch based on GlobalFM mode
    if (mFabricMode == FM_MODE_BAREMETAL) {
        startNonSharedFabricMode();
    } else {
        if ( isFabricModeRestart() == true ) {
            restartSharedFabricMode();
        } else {
            startSharedFabricMode();
        }
    }
    
    if ( !mStagedInit ) {
        finishGlobalFMInitialization();
    }
}

void
GlobalFabricManager::finishGlobalFMInitialization(void)
{
    if (mFabricMode == FM_MODE_SHARED_LWSWITCH)
    {
        // at this stage, some of the cached GPU related data is stale as the GPUs are detached
        // from ServiceVM. So, clearing them to make it explicit.
        mGpuInfoMap.clear();
        mExcludedGpuInfoMap.clear();
        mGpuLWLinkConnMatrix.clear();
        mGpuLWLinkSpeedInfoMap.clear();
    }
    
    // wait and check if config error has oclwrred due to
    // error response or timeout
    if ( waitForAllFabricNodeConfigCompletion() != FM_INT_ST_OK )
    {
        std::ostringstream ss;
        ss << "failed to configure all the available GPUs or LWSwitches" << endl;
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // All our initialization for the node is done. Tell LFM to allow peer access
    // by setting appropriate RM/LWSwitch driver state.
    if ( sendInitDoneToAllFabricNodes() != FM_INT_ST_OK )
    {
        std::ostringstream ss;
        ss << "failed to initialize all the available GPUs or LWSwitches" << endl;
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }
 
    // save fm state only in shared fabric mode without restart
    if (((getFabricMode() == FM_MODE_SHARED_LWSWITCH) || (getFabricMode() == FM_MODE_VGPU)) && (isFabricModeRestart() == false)) {
        mpHaMgr->saveStates();
    }

    //
    // we are pretty much done with all the FM initialization sequence. 
    // create and initialize our library interface sockets
    //
#if !defined(LW_MODS) || defined(LW_MODS_GDM_BUILD)
    createFmApiSocketInterafces();
#endif

    FM_LOG_INFO("Successfully configured all the available GPUs and LWSwitches.");
    FM_SYSLOG_NOTICE("Successfully configured all the available GPUs and LWSwitches.");
#ifndef LW_MODS
    printf("Successfully configured all the available GPUs and LWSwitches.\n");
#endif

    // set shared fabric mode initialization status, to support FM APIs.
    mFabricManagerInitDone = true;
}

/*****************************************************************************/
GlobalFabricManager::~GlobalFabricManager()
{
    cleanup();
};

/*****************************************************************************/
void GlobalFabricManager::cleanup()
{
    //
    // TODO - move this to appropriate place (like error handling) later
    // Tell LFM to stop peer access by setting appropriate RM/LWSwitch driver state.
    // Also we want to keep running LWCA apps when FM exits. So this clean-up has to
    // be conditional.
    //
    //sendDeInitToAllFabricNodes();

    workqueue_shutdown(&mErrWorkQueue);

    if (mGlobalCmdServer) {
        delete mGlobalCmdServer;
    }

    MsgHandlerList::iterator mit = mMsgHandlers.begin();
    while ( mit != mMsgHandlers.end() ) {
        FMMessageHandler* pMsgHandlr = *mit;
        mit = mMsgHandlers.erase( mit );
        delete pMsgHandlr;
    }

    std::map <uint32_t, FMFabricNode*>::iterator it = mvFabricNodes.begin();
    while ( it != mvFabricNodes.end() ) {
        FMFabricNode* pFabricNode = it->second;
        mvFabricNodes.erase( it++ );
        delete pFabricNode;
    }

    if (mGlobalFMLibCmdServer) delete mGlobalFMLibCmdServer;
    if (mTopoValidator) delete mTopoValidator;
    if (mpParser) delete mpParser;
    if (mGfmPartitionMgr) delete mGfmPartitionMgr;
    if (mpHaMgr) delete mpHaMgr;
    if (mpDegradedModeMgr) delete mpDegradedModeMgr;
    if (mGlobalFMInternalCmdServer) delete mGlobalFMInternalCmdServer;
    if (mGlobalFmApiHandler) delete mGlobalFmApiHandler;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    if (mFabricNodeConfigFile) delete mFabricNodeConfigFile;
    if (mMultiNodeTopology) delete mMultiNodeTopology;
#endif
    
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if (mpMcastMgr) delete mpMcastMgr;
#endif

    free(mFmLibCmdBindInterface);
    free(mFmLibCmdSockPath);

    lwosDeleteCriticalSection( &mLock );
}

/*****************************************************************************/

int
GlobalFabricManager::ProcessMessage(uint32 nodeId, lwswitch::fmMessage * pFmMessage, bool &isResponse)
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
        case lwswitch::FM_NODE_GET_GPU_LWLINK_SPEED_INFO_RSP:            
        case lwswitch::FM_NODE_GET_VERSION_INFO_RSP:
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
        case lwswitch::FM_RMAP_TABLE_RSP:
        case lwswitch::FM_RID_TABLE_RSP:
        case lwswitch::FM_RLAN_TABLE_RSP:
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case lwswitch::FM_MCID_TABLE_SET_RSP:
#endif
            mpConfig->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_LWSWITCH_ERROR_FATAL:
        case lwswitch::FM_LWSWITCH_FATAL_ERROR_SCOPE:
        case lwswitch::FM_LWSWITCH_ERROR_NON_FATAL:
        case lwswitch::FM_LWLINK_ERROR_LWSWITCH_RECOVERY:
        case lwswitch::FM_LWLINK_ERROR_GPU_RECOVERY:
        case lwswitch::FM_LWLINK_ERROR_GPU_FATAL:
            mGlobalErrStatusMsgHndl->handleMessage( pFmMessage );
            isResponse = false;
            break;
        case lwswitch::FM_NODE_INFO_ACK:
            mpConfig->handleMessage( pFmMessage );
            isResponse = true;
            break;
        case lwswitch::FM_DEGRADED_GPU_INFO_ACK:
        case lwswitch::FM_DEGRADED_LWSWITCH_INFO_ACK:
            mpDegradedModeMgr->handleMessage( pFmMessage );
            isResponse = true;
            break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case lwswitch::FM_MULTICAST_GROUP_CREATE_REQ:
        case lwswitch::FM_MULTICAST_GROUP_BIND_REQ:
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_REQ:
        case lwswitch::FM_MULTICAST_GROUP_SETUP_COMPLETE_ACK:
        case lwswitch::FM_MULTICAST_GROUP_RELEASE_COMPLETE_ACK:
        {
            mpMcastMgr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }

        case lwswitch::FM_LWLINK_INBAND_MSG:
        {
            // Inband TODO
            // mpLWLinkInbandMsgHndlr->handleMessage(pFmMessage);
            isResponse = false;
            break;
        }
#endif
        default:
            FM_LOG_ERROR("unknown message type %d detected in global fabric manager message handler", pFmMessage->type());
            isResponse = false;
            break;
    }

    lwosLeaveCriticalSection( &mLock );

    return FM_INT_ST_OK;
}

void
GlobalFabricManager::OnFabricNodeConnect(uint32 nodeId)
{
    MsgHandlerList::iterator it;
    for( it = mMsgHandlers.begin(); it != mMsgHandlers.end(); ++it ) {
        FMMessageHandler* msgHdlr = (*it);
        msgHdlr->handleEvent( FM_EVENT_PEER_FM_CONNECT, nodeId );
    }
}

void
GlobalFabricManager::OnFabricNodeDisconnect(uint32 nodeId)
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

    FM_LOG_ERROR("lost socket connection between global and local fabric manager instance with " NODE_ID_LOG_STR " %d", nodeId);
    FM_SYSLOG_ERR("lost socket connection between global and local fabric manager instance with " NODE_ID_LOG_STR " %d", nodeId);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FM_LOG_ERROR("Aborting all the LWCA jobs and setting cluster nodes to uninitialized."
                 " New LWCA job launch will fail on this cluster of nodes."
                 " Refer to system user guide for recovery procedure");
    FM_SYSLOG_ERR("Aborting all the LWCA jobs and setting cluster nodes to uninitialized."
                  " New LWCA job launch will fail on this cluster of nodes."
                  " Refer to system user guide for recovery procedure");
#endif

    // TODO For Front Nine systems all nodes need to close FM session. Re-visit this when
    // reconnection to nodes is fully implemented
    sendDeInitToAllFabricNodes();
}

FMIntReturn_t
GlobalFabricManager::SendMessageToGfm(lwswitch::fmMessage *pFmMessage, bool trackReq)
{
    FM_LOG_DEBUG("SendMessageToGfm within Global Fabric Manager is not supposed to be called\n");
    return FM_INT_ST_NOT_SUPPORTED;
}

FMIntReturn_t
GlobalFabricManager::SendMessageToLfm(uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage,
                                      bool trackReq)
{
    FM_LOG_DEBUG("GlobalFabricManager: SendMessageToLfm ");
    std::map <uint32_t, FMFabricNode *>::iterator it;
    it = mvFabricNodes.find( fabricNodeId );
    if ( it == mvFabricNodes.end() )
    {
        FM_LOG_DEBUG("Invalid " NODE_ID_LOG_STR " %d", fabricNodeId);
        return FM_INT_ST_NOT_CONFIGURED;
    }
 
    FMFabricNode *pFabricNode = it->second;
 
    return pFabricNode->SendControlMessage( pFmMessage, trackReq );
}

FMIntReturn_t
GlobalFabricManager::SendMessageToLfmSync(uint32 fabricNodeId,
                                          lwswitch::fmMessage *pFmMessage,
                                          lwswitch::fmMessage **pResponse,
                                          uint32_t timeoutSec)
{
    FM_LOG_DEBUG("GlobalFabricManager: SendMessageToLfm ");
    std::map <uint32_t, FMFabricNode *>::iterator it;

    it = mvFabricNodes.find( fabricNodeId );
    if ( it == mvFabricNodes.end() )
    {
        FM_LOG_DEBUG("Invalid " NODE_ID_LOG_STR " %d", fabricNodeId);
        return FM_INT_ST_NOT_CONFIGURED;
    }

    FMFabricNode *pFabricNode = it->second;

    return pFabricNode->SendControlMessageSync( pFmMessage, pResponse, timeoutSec );
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
bool
GlobalFabricManager::isMultiNodeMode()
{
    if (mMultiNodeTopology == NULL || strnlen(mMultiNodeTopology, FM_CONFIG_MAX_STRING_ITEM_LEN) == 0) {
        return false;
    } 
    
    // this means that the topologyType was specified by the customer
    // meaning that this is a manual override to enable multi node 
    return true;
}

FMIntReturn_t
GlobalFabricManager::parseMultiNodeFabricTopology(const char *topoFile)
{
    FMIntReturn_t rc;

    // try the topology file at the default location
    rc = mpParser->parseFabricTopology(topoFile);
    if (rc == FM_INT_ST_OK) {
        FM_LOG_INFO("fabric topology file %s is parsed.", topoFile);
        FM_LOG_INFO("number of node: %d, LWSwitch: %d, Port: %d, ReqEntry: %d, RespEntry: %d, GPU: %d",
                   (int)mpParser->NodeCfg.size(), (int)mpParser->lwswitchCfg.size(),
                   (int)mpParser->portInfo.size(), (int)mpParser->reqEntry.size(),
                   (int)mpParser->respEntry.size(), (int)mpParser->gpuCfg.size());
        return rc;
    }

    FM_LOG_ERROR("failed to parse topology file: %s with error: %d.", topoFile, rc);
    return rc;
}
#endif
FMIntReturn_t
GlobalFabricManager::parseNodeFabricTopology(const char *topoFile)
{
    FMIntReturn_t rc;

    rc = mpParser->parseFabricTopology(topoFile);
    if (rc == FM_INT_ST_OK) {
        FM_LOG_INFO("fabric topology file %s is parsed.", topoFile);
        FM_LOG_INFO("number of LWSwitches: %d, Port: %d, request routing entry: %d, response routing entry: %d, GPUs: %d",
                   (int)mpParser->lwswitchCfg.size(),
                   (int)mpParser->portInfo.size(), (int)mpParser->reqEntry.size(),
                   (int)mpParser->respEntry.size(), (int)mpParser->gpuCfg.size());
        return rc;
    }

    FM_LOG_ERROR("failed to parse fabric topology file: %s with error: %d.", topoFile, rc);
    return rc;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
/******************************************************************************************
* Multinode IP address and Node ID information file is a text file with following format

*     <IP address of nodeX>  <node ID of nodeX>
*     <IP address of nodeY>  <node ID of nodeY>
*     Also, any lines starting with a # will be treated as a comment

* Note: We expect this file to be filled by the cluster admin. So, only a minimal validation
* is done during parsing.
*******************************************************************************************/
void
GlobalFabricManager::readFabricNodeConfigFile(char *fabricNodeConfigFile,
                                               std::map<uint32_t, std::string> &nodeToIpAddrMap)
{
    std::ifstream addrFileStream(fabricNodeConfigFile, std::ifstream::in);

    if (!addrFileStream.is_open()) {
        std::ostringstream ss;
        ss << "request to open fabric node IP address information file " << fabricNodeConfigFile << " failed";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str().c_str());
    }

    std::string lineBuf;
    while (std::getline(addrFileStream, lineBuf)) {
        // first remove trailing white spaces
        lineBuf.erase(0, lineBuf.find_first_not_of(" \t"));
        // check for empty line
        if (lineBuf.empty()) {
            continue;
        }
        // treat line starting with "#" as comment
        if ('#' == lineBuf.front()) {
            continue;
        }

        // a good line, parse the ip and node address info
        // construct a stream from the string for tokenizing 
        std::stringstream lineStream(lineBuf);
        std::istream_iterator<std::string> it(lineStream);
        std::istream_iterator<std::string> end;
        std::vector<std::string> strTokens(it, end);
        // we are expecting only two tokens per line (ip address and node id)
        if (2 != strTokens.size()) {
            std::ostringstream ss;
            ss << "fabric node IP address information file contains unexpected line: " << lineBuf ;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str().c_str());
        }
        // check whether the specified node address is a valid ip address by doing a inet_pton colwersion
        struct sockaddr_in tempSockAddr;
        memset(&tempSockAddr, 0, sizeof(tempSockAddr));
        std::string nodeAddrStr = strTokens[0];
        if (1 != evutil_inet_pton(AF_INET, nodeAddrStr.c_str(), &tempSockAddr.sin_addr)) {
            std::ostringstream ss;
            ss << "fabric node IP address information file contains invalid " NODE_ID_LOG_STR " address line: " << lineBuf;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str().c_str());
        }
        // check whether node id is a valid numeric
        std::string nodeIdStr = strTokens[1];
        char* endPtr = NULL;
        int nodeId = std::strtol(nodeIdStr.c_str(), &endPtr, 10);
        if (*endPtr) {
            std::ostringstream ss;
            ss << "fabric node IP address information file contains invalid " NODE_ID_LOG_STR " line: " << lineBuf;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str().c_str());
        }
        // all looks good, add this ip address and node id into the address map
        nodeToIpAddrMap[nodeId] = nodeAddrStr;
    }

    addrFileStream.close();

    // dump parsed node id and ip address information to log file
    std::ostringstream outStr;
    outStr << "dumping all the parsed node ip address and " NODE_ID_LOG_STR " information" << std::endl;
    std::map<uint32_t, std::string>::iterator it;
    for (it = nodeToIpAddrMap.begin(); it != nodeToIpAddrMap.end(); it++) {
        outStr << "  IP Address: " << it->second << " " << NODE_ID_LOG_STR << " " << it->first << std::endl;
    }
    FM_LOG_INFO("%s", outStr.str().c_str());
}
#endif

/*****************************************************************************************************
 This function will create Fabric Nodes instance to establish connection to Local Fabric Managers.
 Used only in multi-node mode
*****************************************************************************************************/
void
GlobalFabricManager::createFabricNodes(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    FMFabricNode *pFabricNode;

    mvFabricNodes.clear();

    for ( it = mpParser->NodeCfg.begin();
          it != mpParser->NodeCfg.end(); it++ )
    {
        pNode = it->second;
        pFabricNode =  new FMFabricNode(pNode->IPAddress->c_str(),
                                        pNode->nodeId, this, false);

        // keep the node in our global list
        mvFabricNodes.insert( std::make_pair(pNode->nodeId, pFabricNode) );
    }
}

/********************************************************************************************************
 This function will create single Fabric Node instance to establish connection to Local Fabric Managers.
 Used only in single-node mode
*********************************************************************************************************/
void
GlobalFabricManager::createFabricNode(uint32_t nodeId, char *localFMIPAddress, char *domainSocketPath)
{
    FMFabricNode *pFabricNode;

    mvFabricNodes.clear();
    // use domain socket or local FM's IP address for connection
    if (domainSocketPath == NULL || !(strnlen(domainSocketPath, FM_CONFIG_MAX_STRING_ITEM_LEN))) {
        pFabricNode =  new FMFabricNode(localFMIPAddress, nodeId, this, false);
    } else {
        pFabricNode =  new FMFabricNode(domainSocketPath, nodeId, this, true);
    }

    // keep the node in our global list
    mvFabricNodes.insert( std::make_pair(nodeId, pFabricNode) );
}

bool
GlobalFabricManager::waitForAllFabricNodeCtrlConns(void)
{
    std::map <uint32_t, FMFabricNode *>::iterator it;
    std::map <uint32_t, FMFabricNode*> tempFabricNodes;

    // make a copy of the nods
    tempFabricNodes = mvFabricNodes;

    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    timelib64_t lastErrorLogTime = timeStart;
    const unsigned int MICRO_SECONDS_TO_SEC = 1000000;
#endif    
    const unsigned int WAIT_MS = 50; // wait interval
    timelib64_t timeoutUs = mGfmWaitTimeout * 1000ULL * 1000ULL; // total time wait for the connection to establish

    // iterate over all the available nodes for connection
    while (true) {
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end();) {
            FMFabricNode *pFabricNode = it->second;
            if (pFabricNode->isControlConnectionActive()) {
                tempFabricNodes.erase(it++);
            } else {
                ++it;
            }
         }

        // one iteration is passed. exit if don't have any more nodes pending
        timeNow = timelib_usecSince1970();
        if (tempFabricNodes.empty()) {
            break;
        } else if (mGfmWaitTimeout >= 0) {
            if ((timeNow - timeStart) > timeoutUs) {
                // elapsed all the time and still there are unconnected nodes.
                break;
            }
        }
        // wait for some time and poll nodes for connection again
        lwosSleep(WAIT_MS);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        //
        // after few meaningful iterations, if the connection is not established, start
        // logging the connection failure, so user knows what is happening.
        //
        // print a warning message every 3 second
        //
        if ((timeNow - lastErrorLogTime) > (MICRO_SECONDS_TO_SEC *3)) {
            lastErrorLogTime = timeNow;
            for (it = tempFabricNodes.begin(); it != tempFabricNodes.end(); it++) {
                FMFabricNode *pFabricNode = it->second;
                std::ostringstream strFailedNodes;
                strFailedNodes << "waiting to establish fabric manager control connection with NodeID:" << pFabricNode->getNodeId()
                               << " IP Address:" << pFabricNode->getNodeAddress() << std::endl;
                FM_LOG_WARNING("%s", strFailedNodes.str().c_str());
                FM_SYSLOG_WARNING("%s", strFailedNodes.str().c_str());
            }
        }
#endif
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
GlobalFabricManager::configureAllTrunkPorts(void)
{
    std::set<enum PortType> portTypes;

    portTypes.insert( TRUNK_PORT_SWITCH );
    mpConfig->configSwitchPortsWithTypes( portTypes );
}

void 
GlobalFabricManager::checkFabricNodeVersions(void)
{
    bool mismatch = false;
    mismatch = GFMHelper::getFMVersionInfoForAllNodes(mpParser, mDevInfoMsgHndlr);

    if (mismatch) {
        throw std::runtime_error("version mismatch was detected between global fabric manager and" 
              "local fabric manager instances on certain fabric nodes");
    }
}

void
GlobalFabricManager::sendGlobalConfigToAllFabricNodes(void)
{
    std::map <uint32_t, FMFabricNode*>::iterator it;

    // iterate through all the fabric node objects and send global config 
    for (it = mvFabricNodes.begin(); it != mvFabricNodes.end(); it++) {
        uint32_t nodeId = it->first;
        // send nodeId and other information for the node
        mpConfig->sendNodeGlobalConfig( nodeId );
    }

    // Note: on multi-node system, wait for all the peer LFM connections to 
    // establish by checking for response messages
}

void
GlobalFabricManager::sendPeerLFMInfoToAllFabricNodes(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;

    for ( it = mpParser->NodeCfg.begin();
        it != mpParser->NodeCfg.end(); it++ )
    {
        pNode = it->second;
        // send all the peer nodes information (IP,NodeID etc) so that
        // peer LFM connection is established before link training.
        mpConfig->sendPeerLFMInfo( pNode->nodeId );
    }

    // Note: on multi-node system, wait for all the peer LFM connections to 
    // establish by checking for response messages
}

void
GlobalFabricManager::configureAllFabricNodes(void)
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
            FM_LOG_INFO("no LWSwitches detected for " NODE_ID_LOG_STR " %d, skipping LWSwitch and GPU configuration.", pNode->nodeId);
            continue;
        }

        // configure access ports only here,
        // as trunk ports are already configured before link training
        portTypes.insert( ACCESS_PORT_GPU );
        FMIntReturn_t rc = mpConfig->configOneNode(pNode->nodeId, portTypes);
        if ( rc != FM_INT_ST_OK )
        {
            // set node config error state
            setNodeConfigError(pNode->nodeId);
        }
     }
}

FMIntReturn_t
GlobalFabricManager::waitForAllFabricNodeConfigCompletion(void)
{
    // wait for all nodes finish processing configuration
    std::map <uint32_t, FMFabricNode*>::iterator it;
    std::map <uint32_t, FMFabricNode*> tempFabricNodes;
    FMFabricNode *pFabricNode;

    // make a copy of the nodes
    tempFabricNodes = mvFabricNodes;

    timelib64_t timeStart = timelib_usecSince1970();
    timelib64_t timeNow = timeStart;
    const unsigned int WAIT_MS = 50; // wait interval
    unsigned int timeoutMs = mGfmWaitTimeout * 1000; // total time wait for the configuration to establish

    // iterate over all the available nodes
    while (true) {
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end();) {
            pFabricNode = it->second;

            // check if there is any config error has happened
            if ( isNodeConfigErrorOclwred(pFabricNode->getNodeId()) == true )
            {
                FM_LOG_ERROR("failed to configure local fabric manager " NODE_ID_LOG_STR " %d", pFabricNode->getNodeId());
                return FM_INT_ST_CFG_ERROR;
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
        } else if (mGfmWaitTimeout > 0) { //negative gfm wait timeout denotes infinite wait time
            timeNow = timelib_usecSince1970();
            if ((timeNow - timeStart) + WAIT_MS*1000 > timeoutMs*1000) {
                // elapsed all the time and still there are unfinished nodes.
                break;
            }
        }
        // wait for some time
         lwosSleep(WAIT_MS);
    }

    if (tempFabricNodes.size()) {
        // some nodes have not finished configuration
        FM_LOG_ERROR("not all local fabric manager instances finished their configuration");
        for (it = tempFabricNodes.begin(); it != tempFabricNodes.end(); it++) {
            pFabricNode = it->second;

            lwswitch::fmMessage errMsg;
            mpConfig->handleConfigError( pFabricNode->getNodeId(),
                                         ERROR_SOURCE_SW_GLOBALFM,
                                         ERROR_TYPE_CONFIG_TIMEOUT,
                                         errMsg );
        }
        return FM_INT_ST_CFG_TIMEOUT;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
GlobalFabricManager::sendInitDoneToAllFabricNodes(void)
{
    FMIntReturn_t rc;

    // wait for all nodes finish processing configuration
    std::map <uint32_t, FMFabricNode*>::iterator it;
    FMFabricNode *pFabricNode;

    // send init done to all nodes, because all nodes have finished configuration
    for (it = mvFabricNodes.begin(); it != mvFabricNodes.end(); it++) {
        pFabricNode = it->second;
        rc = mpConfig->sendConfigInitDoneReqMsg(pFabricNode->getNodeId());
        if ( rc != FM_INT_ST_OK )
        {
            // TODO in multi node needs to consider each node separately
            return rc;
        }
    }
    return FM_INT_ST_OK;
}

void
GlobalFabricManager::sendDeInitToAllFabricNodes(void)
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
GlobalFabricManager::dumpAllGpuInfo(void)
{
    FMGpuInfoMap::iterator it;
    std::stringstream outStr;

    for ( it = mGpuInfoMap.begin(); it != mGpuInfoMap.end(); it++ ) {
        outStr << "dumping all the detected GPU information" << std::endl;
        FMGpuInfoList gpuList = it->second;
        FMGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            FMGpuInfo_t gpuInfo = (*jit);
            outStr << "  Index: " << std::setfill('0') << std::setw(2) << int(gpuInfo.gpuIndex);
            outStr << "  PCI Bus ID: " << gpuInfo.pciInfo.busId;
            outStr << "  Discovered Link Mask: " << std::hex << gpuInfo.discoveredLinkMask;
            outStr << "  Enabled Link Mask: " << std::hex << gpuInfo.enabledLinkMask;
            outStr << "  UUID: " << gpuInfo.uuid.bytes << std::endl;
        }
        // add two new line for next node
        outStr << std::endl << std::endl;
    }

    FMExcludedGpuInfoMap::iterator blit;
 
    for ( blit = mExcludedGpuInfoMap.begin(); blit != mExcludedGpuInfoMap.end(); blit++ ) {
        outStr << "dumping all the excluded GPU information" << std::endl;
        FMExcludedGpuInfoList gpuList = blit->second;
        FMExcludedGpuInfoList::iterator jit;
        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            FMExcludedGpuInfo_t gpuInfo = (*jit);
            // no enumeration index for excluded GPUs
            outStr << "  PCI Bus ID: " << gpuInfo.pciInfo.busId;
            outStr << "  UUID: " << gpuInfo.uuid.bytes << std::endl;
            // log excluded gpu uuid information to syslog
            FM_SYSLOG_NOTICE("GPU pci bus id: %s uuid: %s is excluded", gpuInfo.pciInfo.busId, gpuInfo.uuid.bytes);
        }
    }

    std::string strInfo = outStr.str();
    if (strInfo.size() == 0) {
        strInfo = "No GPU information is available to dump\n";
    }

    FM_LOG_INFO("%s", strInfo.c_str());
}

void
GlobalFabricManager::dumpAllLWSwitchInfo(void)
{
    FMLWSwitchInfoMap::iterator it;
    FMLWSwitchInfoMap switchInfoMap = mLwswitchInfoMap;
    std::stringstream outStr;

    for ( it = switchInfoMap.begin(); it != switchInfoMap.end(); it++ ) {
        outStr << "dumping all the detected LWSwitch information" << std::endl;
        FMLWSwitchInfoList switchList = it->second;
        FMLWSwitchInfoList::iterator jit;
        for ( jit = switchList.begin(); jit != switchList.end(); jit++ ) {
            FMLWSwitchInfo switchInfo = (*jit);
            outStr << "  Index: " << std::setfill('0') << std::setw(2) << int(switchInfo.switchIndex);
            outStr << "  Physical Id: " << std::dec << switchInfo.physicalId;
            outStr << "  PCI Bus ID: " << switchInfo.pciInfo.busId;
            outStr << "  Enabled Link Mask: " << std::hex << switchInfo.enabledLinkMask;
            outStr << "  Arch Type: " << int(switchInfo.archType);
            outStr << "  UUID : " << switchInfo.uuid.bytes << std::endl;
        }
        // add two new line for next node
        outStr << std::endl << std::endl;
    }

    FMExcludedLWSwitchInfoMap::iterator bit;
    FMExcludedLWSwitchInfoMap excludedSwitchInfoMap = mExcludedLwswitchInfoMap;

    for (bit = excludedSwitchInfoMap.begin(); bit != excludedSwitchInfoMap.end(); bit++) {
        outStr << "dumping all excluded LWSwitch information " << std::endl;
        FMExcludedLWSwitchInfoList excludedSwitchInfoList = bit->second;
        FMExcludedLWSwitchInfoList::iterator bjit;
        for (bjit = excludedSwitchInfoList.begin(); bjit != excludedSwitchInfoList.end(); bjit++) {
            FMExcludedLWSwitchInfo_t excludedSwitchInfo = (*bjit);
            outStr << "  Physical Id: " << std::setfill('0') << std::setw(2) << int(excludedSwitchInfo.physicalId);
            outStr << "  PCI Bus ID: " << excludedSwitchInfo.pciInfo.busId<< std::endl;
        }
        // add two new line for next node
        outStr << std::endl << std::endl;
    }

    std::string strInfo = outStr.str();
    if (strInfo.size() == 0) {
        strInfo = "No LWSwitch information is available to dump\n";
    }

    FM_LOG_INFO("%s", strInfo.c_str());
}

void
GlobalFabricManager::dumpLWLinkDeviceAndInitStatus(void)
{
    std::map <NodeKeyType, NodeConfig *>::iterator it;
    NodeConfig *pNode;
    for ( it = mpParser->NodeCfg.begin(); it != mpParser->NodeCfg.end(); it++ ) {
        pNode = it->second;
        FM_LOG_INFO("\ndumping GPU and LWSwitch LWLink information for " NODE_ID_LOG_STR " %d", pNode->nodeId);
        FMLWLinkDevInfoList devList;
        mLWLinkDevRepo.getDeviceList( pNode->nodeId, devList );
        FMLWLinkDevInfoList::iterator jit;
        for ( jit = devList.begin(); jit != devList.end(); jit++ ) {
            FMLWLinkDevInfo devInfo = (*jit);
            std::stringstream outStr;
            outStr << std::endl; // for pretty printing in our log
            devInfo.dumpInfo( &outStr );
            FM_LOG_INFO("%s", outStr.str().c_str());
        }
    }
}

void
GlobalFabricManager::dumpGpuConnMatrix(void)
{
    FMGpuLWLinkConnMatrixMap::iterator it;
    std::stringstream outStr;

    outStr << "dumping callwlated GPU enumeration index to physical id mapping info" << std::endl;
    outStr << "GPU LWLink connection and ID mapping matrix size = " << mGpuLWLinkConnMatrix.size() << std::endl;

    for (it = mGpuLWLinkConnMatrix.begin(); it != mGpuLWLinkConnMatrix.end(); it++) {
        FMGpuLWLinkConnMatrixList connMatrixList = it->second;
        FMGpuLWLinkConnMatrixList::iterator jit;
        for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
            FMGpuLWLinkConnMatrix gpuConnInfo = *jit;
            outStr << "GPU Physical Id = " << gpuConnInfo.gpuPhyIndex;
            outStr << " Enum Index = " << gpuConnInfo.gpuEnumIndex;
            outStr << " UUID = " << gpuConnInfo.uuid.bytes;
            for ( uint32_t idx = 0; idx < FMDeviceProperty::getLWLinksPerGpu(mSwitchArchType); idx++ ) {
                outStr << " LinkActive[" << idx <<"]= " << gpuConnInfo.linkConnStatus[idx];
            }
            outStr << std::endl;
        }
    }

    FM_LOG_INFO("%s", outStr.str().c_str());
}

void
GlobalFabricManager::queueErrorWorkerRequest(uint32 nodeId, lwswitch::fmMessage *errorMsg)
{
    // As part of error handling, we may send a re-train/FM session clear request
    // to corresponding LFM and wait for it to complete. But we are lwrrently in the context
    // of LibEvent thread ie FmClientListener::ReadCB() which triggered the process message.
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

    pJob->job_function = GlobalFabricManager::processErrorWorkerRequest;
    pJob->user_data = pWorkerReqInfo;
    workqueue_add_job(&mErrWorkQueue, pJob);
}

void
GlobalFabricManager::processErrorWorkerRequest(job_t *pJob)
{
    FmErrorWorkerReqInfo *pWorkerReqInfo;
    pWorkerReqInfo = (FmErrorWorkerReqInfo*) pJob->user_data;
    GlobalFabricManager *pGfmObj = pWorkerReqInfo->pGfmObj;
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
        case lwswitch::FM_LWSWITCH_FATAL_ERROR_SCOPE: {
            errSource = ERROR_SOURCE_LWSWITCH;
            errType = ERROR_TYPE_LWSWITCH_FATAL_SCOPE;
            break;
        }
        default: {
            FM_LOG_ERROR("unknown GPU or LWSwitch error type information in error worker handler");
            delete pWorkerReqInfo;
            delete pJob;
            return;
        }
    }

    // ilwoke the actual error handler
    GlobalFMErrorHndlr errHndlr(pGfmObj, nodeId, 0, errSource, errType, errorMsg);
    errHndlr.processErrorMsg();

    delete pWorkerReqInfo;
    delete pJob;
}

bool
GlobalFabricManager::getGpuPhysicalIndex(uint32 nodeId, uint32_t enumIndex, uint32_t &physicalIdx)
{
    FMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        FMGpuLWLinkConnMatrix gpuConnInfo = *jit;
        if (gpuConnInfo.gpuEnumIndex == enumIndex) {
            physicalIdx = gpuConnInfo.gpuPhyIndex;
            return true;
        }
    }

    return false;
}

bool
GlobalFabricManager::getGpuEnumIndex(uint32 nodeId, uint32_t physicalIdx, uint32_t &enumIndex)
{
    FMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for ( jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++ ) {
        FMGpuLWLinkConnMatrix gpuConnInfo = *jit;
        if (gpuConnInfo.gpuPhyIndex == physicalIdx) {
            enumIndex = gpuConnInfo.gpuEnumIndex;
            return true;
        }
    }

    return false;
}

bool
GlobalFabricManager::getGpuPciBdf(uint32 nodeId, uint32_t enumIndex, FMPciInfo_t &pciInfo)
{
    FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    FMGpuInfoList::iterator it;
    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        FMGpuInfo_t tempInfo = (*it);
        if (tempInfo.gpuIndex == enumIndex) {
            pciInfo = tempInfo.pciInfo;
            return true;
        }
    }

    // not found the pci bdf
    return false;
}

bool
GlobalFabricManager::getGpuPciBdf(uint32 nodeId, char uuid[], FMPciInfo_t &pciInfo)
{
    FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    FMGpuInfoList::iterator it;
    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        FMGpuInfo_t tempInfo = (*it);
        if (strncmp(tempInfo.uuid.bytes, uuid, FM_UUID_BUFFER_SIZE) == 0) {
            pciInfo = tempInfo.pciInfo;
            return true;
        }
    }

    // not found the pci bdf
    return false;
}

bool
GlobalFabricManager::getGpuUuid(uint32 nodeId, uint32 physicalId, char uuid[])
{
    FMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return false;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for (jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++) {
        FMGpuLWLinkConnMatrix gpuConnInfo = (*jit);
        if (gpuConnInfo.gpuPhyIndex == physicalId) {
            strncpy(uuid, gpuConnInfo.uuid.bytes, FM_UUID_BUFFER_SIZE - 1);
            return true;
        }
    }

    // not found the uuid
    return false;
}

bool
GlobalFabricManager::getGpuPhysicalId(uint32 nodeId, char* uuid, uint32_t &physicalId)
{
    uint32_t isFound = false;
    FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    FMGpuInfoList::iterator gpuInfoIt;
    FMGpuInfo_t *gpuInfo = NULL;

    for (gpuInfoIt = gpuList.begin(); gpuInfoIt != gpuList.end(); gpuInfoIt++) {
        FMGpuInfo_t *tempInfo = &(*gpuInfoIt);

        if (strncmp(tempInfo->uuid.bytes, uuid, FM_UUID_BUFFER_SIZE) == 0) {
            gpuInfo = tempInfo;
            break;
        }
    }

    // uuid is not found
    if (gpuInfo == NULL) {
        return isFound;
    }

    // use physicalId to enumIdx mapping
    FMGpuLWLinkConnMatrixMap::iterator it = mGpuLWLinkConnMatrix.find(nodeId);
    if (it == mGpuLWLinkConnMatrix.end()) {
        // no entry for the specified node
        return isFound;
    }

    FMGpuLWLinkConnMatrixList connMatrixList = it->second;
    FMGpuLWLinkConnMatrixList::iterator jit;
    for (jit = connMatrixList.begin(); jit != connMatrixList.end(); jit++) {
        FMGpuLWLinkConnMatrix gpuConnInfo = (*jit);
        if (gpuConnInfo.gpuEnumIndex == gpuInfo->gpuIndex) {
            isFound = true;
            physicalId = gpuConnInfo.gpuPhyIndex;
            break;
        }
    }

    return isFound;
}

bool
GlobalFabricManager::getGpuNumActiveLWLinks(uint32 nodeId, FMPciInfo_t pciInfo, uint32_t &numActiveLinks)
{
    //
    // GPU & LWSwitch device link state is updated after training and kept in FMLWLinkDevInfo
    // which is from LWLinkCoreLib
    //

    // lookup on devices returned by LWLinkCoreLib driver.
    FMLWLinkDevInfo lwLinkDevInfo;
    if (!mLWLinkDevRepo.getDeviceInfo(nodeId, pciInfo, lwLinkDevInfo)) {
        // not able to find a matching GPU. this shouldn't happen
        return false;
    }

    // found the GPU device. get its active LWLink count
    return lwLinkDevInfo.getNumActiveLinks(numActiveLinks);
}

bool
GlobalFabricManager::getGpuLinkSpeedInfo(uint32 nodeId, char uuid[], FMLWLinkSpeedInfo &linkSpeedInfo)
{
    FMGpuLWLinkSpeedInfoMap::iterator it = mGpuLWLinkSpeedInfoMap.find(nodeId);
    if (it == mGpuLWLinkSpeedInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    FMGpuLWLinkSpeedInfoList speedInfoList = it->second;
    FMGpuLWLinkSpeedInfoList::iterator jit = speedInfoList.begin();
    FMUuid_t uuidInfo = {{0}};
    strncpy(uuidInfo.bytes, (char *)uuid, FM_UUID_BUFFER_SIZE - 1);

    // find the desired GPU
    for (jit = speedInfoList.begin(); jit != speedInfoList.end(); jit++) {
        FMGpuLWLinkSpeedInfo gpuLinkSpeedInfo = (*jit);
        if (gpuLinkSpeedInfo.uuid == uuidInfo) {
            // copying only first link's speed as lwrrently the speed is uniform
            linkSpeedInfo = gpuLinkSpeedInfo.linkSpeedInfo.front();
            return true;
        }
    }

    // not found the desired GPU or GPU list is empty.
    return false;
}


bool
GlobalFabricManager::getGpuDiscoveredLinkMask(uint32 nodeId, uint32_t enumIndex, uint32_t &discoveredLinkMask)
{
    FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    FMGpuInfoList::iterator it;
    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        FMGpuInfo_t tempInfo = (*it);
        if (tempInfo.gpuIndex == enumIndex) {
            discoveredLinkMask = tempInfo.discoveredLinkMask;
            return true;
        }
    }

    // specified GPU is not found
    return false;
}

/*************************************************************************************
 * For a given LWSwitch Physical ID, find the corresponding device's ID in the
 * LWLinkCoreLib driver. LWSwitch driver and CoreLib driver uses different IDs and 
 * the common factor between them is PCI BDF. CoreLib driver's device IDs are derived
 * from PCI BDF.
**************************************************************************************/
bool
GlobalFabricManager::getLWSwitchLWLinkDriverId(uint32 nodeId, uint32 physicalId, uint64 &lwLinkSwitchId)
{
    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);

    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    // find the corresponding LWSwitch's PCI BDF
    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    FMLWSwitchInfo switchInfo;
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
    FMLWLinkDevInfo lwLinkDevInfo;
    if (mLWLinkDevRepo.getDeviceInfo(nodeId, switchInfo.pciInfo, lwLinkDevInfo)) {
        lwLinkSwitchId = lwLinkDevInfo.getDeviceId();
        return true;
    }

    return false;
}

bool
GlobalFabricManager::getLWSwitchPciBdf(uint32 nodeId, uint32 physicalId, FMPciInfo_t &pciInfo)
{
    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);
    
    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    // find the corresponding LWSwitch
    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    FMLWSwitchInfo switchInfo;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        switchInfo = (*jit);
        if (switchInfo.physicalId == physicalId ) {
            // found the desired LWSwitch
            pciInfo = switchInfo.pciInfo;
            return true;
        }
    }

    // default case, not found
    return false;
}

bool
GlobalFabricManager::getExcludedLWSwitchPciBdf(uint32 nodeId, uint32 physicalId, FMPciInfo_t &pciInfo)
{
    FMExcludedLWSwitchInfoMap::iterator bit = mExcludedLwswitchInfoMap.find(nodeId);
    
    if (bit == mExcludedLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    // find the corresponding LWSwitch
    FMExcludedLWSwitchInfoList excludedSwitchList = bit->second;
    FMExcludedLWSwitchInfoList::iterator bjit;
    FMExcludedLWSwitchInfo_t excludedSwitchInfo;
    for (bjit = excludedSwitchList.begin(); bjit != excludedSwitchList.end(); bjit++) {
        excludedSwitchInfo = (*bjit);
        if (excludedSwitchInfo.physicalId == physicalId ) {
            // found the desired LWSwitch
            pciInfo = excludedSwitchInfo.pciInfo;
            return true;
        }
    }

    // default case, not found
    return false;
}

/*************************************************************************************
 * For a given LWSwitch Physical ID, find the corresponding switchInfo.
**************************************************************************************/
bool
GlobalFabricManager::getLWSwitchInfo(uint32 nodeId, uint32 physicalId, FMLWSwitchInfo &switchInfo)
{
    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);

    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    // find the corresponding LWSwitch
    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    FMLWSwitchInfo tmpSwitchInfo;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        tmpSwitchInfo = (*jit);
        if (tmpSwitchInfo.physicalId == physicalId ) {
            // found the matching LWSwitch
            switchInfo = tmpSwitchInfo;
            return true;
        }
    }

    return false;
}

bool
GlobalFabricManager::getLWSwitchPhysicalId(uint32 nodeId, lwlink_pci_dev_info pciDevInfo, uint32 &physicalId)
{
    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);
    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }
    FMLWSwitchInfoList switchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    FMLWSwitchInfo tmpSwitchInfo;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        tmpSwitchInfo = (*jit);
        if (tmpSwitchInfo.pciInfo.domain == pciDevInfo.domain && 
            tmpSwitchInfo.pciInfo.bus == pciDevInfo.bus && 
            tmpSwitchInfo.pciInfo.device == pciDevInfo.device &&
            tmpSwitchInfo.pciInfo.function == pciDevInfo.function) 
        {
            // found the matching LWSwitch
            physicalId = tmpSwitchInfo.physicalId;
            return true;
        }
    }
    return false;
}

/*************************************************************************************
 * For a given GPU physicalId, find the corresponding device's ID in the
 * LWLinkCoreLib driver. LWSwitch driver and CoreLib driver uses different IDs and
 * the common factor between them is PCI BDF. CoreLib driver's device IDs are derived
 * from PCI BDF.
**************************************************************************************/
bool
GlobalFabricManager::getGpuLWLinkDriverId(uint32 nodeId, uint32_t physicalId, uint64_t &lwLinkGpuId)
{
    uint32_t enumIndex;
    if ( getGpuEnumIndex(nodeId, physicalId, enumIndex) == false )
    {
        return false;
    }

    FMPciInfo_t pciInfo;
    if ( getGpuPciBdf(nodeId, enumIndex, pciInfo) == false )
    {
        // pciInfo is not found
        return false;
    }

    // we now have the required GPU PCI BDF, using that
    // do the lookup on devices returned LWLinkCoreLib driver.
    FMLWLinkDevInfo lwLinkDevInfo;
    if (mLWLinkDevRepo.getDeviceInfo(nodeId, pciInfo, lwLinkDevInfo)) {
        lwLinkGpuId = lwLinkDevInfo.getDeviceId();
        return true;
    }

    return false;
}

/*************************************************************************************
 * For a given GPU UUID, find the corresponding device's ID in the
 * LWLinkCoreLib driver. LWSwitch driver and CoreLib driver uses different IDs and
 * the common factor between them is PCI BDF. CoreLib driver's device IDs are derived
 * from PCI BDF.
**************************************************************************************/
bool
GlobalFabricManager::getGpuLWLinkDriverId(uint32 nodeId, char uuid[], uint64_t &lwLinkGpuId)
{
    if ( !uuid )
    {
        return false;
    }

    FMPciInfo_t pciInfo;
    if ( getGpuPciBdf(nodeId, uuid, pciInfo) == false )
    {
        // pciInfo is not found
        return false;
    }

    // we now have the required GPU PCI BDF, using that
    // do the lookup on devices returned LWLinkCoreLib driver.
    FMLWLinkDevInfo lwLinkDevInfo;
    if (mLWLinkDevRepo.getDeviceInfo(nodeId, pciInfo, lwLinkDevInfo)) {
        lwLinkGpuId = lwLinkDevInfo.getDeviceId();
        return true;
    }

    return false;
}

/*
 This function builds a map of all the connected switches 
 for each excluded switch. This is done because if a switch is 
 excluded we prune that switch and hence cannot use the FmFabricParser
 to query info about the excluded switch's peer switches (required
 for degraded mode). This map helps us to find the peers of excluded 
 switches. 
*/
void
GlobalFabricManager::buildConnectedSwitchMappingForExcludedSwitches()
{
    mExcludedToConnectedSwitchInfoMap.clear();
    FMExcludedLWSwitchInfoMap::iterator it;
    for (it = mExcludedLwswitchInfoMap.begin(); it != mExcludedLwswitchInfoMap.end(); it++) {
        uint32_t nodeId = it->first;
        FMExcludedLWSwitchInfoList excludedSwitchInfo = it->second;
        FMExcludedLWSwitchInfoList::iterator sit;
        for (sit = excludedSwitchInfo.begin(); sit != excludedSwitchInfo.end(); sit++) {
            SwitchKeyType switchKey;
            FMExcludedLWSwitchInfo_t excludedSwitch = *sit;
            switchKey.nodeId = nodeId;
            switchKey.physicalId = excludedSwitch.physicalId;
            ConnectedSwitchesInfoMap excludedConnectedSwitches;
            mpParser->getConnectedSwitches( switchKey, excludedConnectedSwitches );
            mExcludedToConnectedSwitchInfoMap.insert(make_pair(switchKey, excludedConnectedSwitches));
        }
    }
}

/*
    This function gets all the partitions for the excluded switch.
    This would be needed because the partitions with excluded switch
    would be pruned and by the time the degraded mode logic runs, if 
    the config option is to disable partitions containing excluded switch
    we would not be able to obtain the info. Hence storing it before pruning.
*/
void
GlobalFabricManager::buildPartitionsWithExcludedSwitch()
{
    mExcludedSwitchPartitionInfoMap.clear();
    FMExcludedLWSwitchInfoMap::iterator it;
    for (it = mExcludedLwswitchInfoMap.begin(); it != mExcludedLwswitchInfoMap.end(); it++) {
        uint32_t nodeId = it->first;
        FMExcludedLWSwitchInfoList excludedSwitchInfo = it->second;
        FMExcludedLWSwitchInfoList::iterator sit;
        for (sit = excludedSwitchInfo.begin(); sit != excludedSwitchInfo.end(); sit++) {
            SwitchKeyType switchKey;
            FMExcludedLWSwitchInfo_t excludedSwitch = *sit;
            switchKey.nodeId = nodeId;
            switchKey.physicalId = excludedSwitch.physicalId;
            PartitionSet partitions;
            mpParser->getSharedLWSwitchPartitionsWithSwitch( switchKey, partitions);
            mExcludedSwitchPartitionInfoMap.insert(make_pair(switchKey, partitions));
        }
    }
}

void 
GlobalFabricManager::getExcludedConnectedSwitches( SwitchKeyType key, ConnectedSwitchesInfoMap &excludedConnectedSwitches)
{
    std::map<SwitchKeyType, ConnectedSwitchesInfoMap>::iterator it = mExcludedToConnectedSwitchInfoMap.find(key);
    if (it == mExcludedToConnectedSwitchInfoMap.end()) {
        return;
    }

    excludedConnectedSwitches.insert(it->second.begin(), it->second.end());
}

void
GlobalFabricManager::getPartitionSetForExcludedSwitch(SwitchKeyType key, PartitionSet &partitions)
{
    std::map<SwitchKeyType, PartitionSet>::iterator it = mExcludedSwitchPartitionInfoMap.find(key);
    if (it == mExcludedSwitchPartitionInfoMap.end()) {
        return;
    }

    partitions.insert(it->second.begin(), it->second.end());
}

bool
GlobalFabricManager::isLwswitchExcluded( uint32 nodeId, uint32_t physicalId )
{
    FMExcludedLWSwitchInfoMap::iterator it = mExcludedLwswitchInfoMap.find(nodeId);

    if (it == mExcludedLwswitchInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    FMExcludedLWSwitchInfoList switchList = it->second;
    FMExcludedLWSwitchInfoList::iterator jit;
    FMExcludedLWSwitchInfo_t switchInfo;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        switchInfo = (*jit);
        if (switchInfo.physicalId == physicalId ) {
            // found the switch in the excluded list
            return true;
        }
    }

    // the switch is not found in the excluded list
    return false;
}

bool
GlobalFabricManager::isGpuExcluded( uint32 nodeId, char uuid[] )
{
    FMExcludedGpuInfoMap::iterator it = mExcludedGpuInfoMap.find(nodeId);

    if (it == mExcludedGpuInfoMap.end()) {
        // no entry for the specified node
        return false;
    }

    FMExcludedGpuInfoList gpuList = it->second;
    FMExcludedGpuInfoList::iterator jit;
    FMExcludedGpuInfo_t gpuInfo;
    for (jit = gpuList.begin(); jit != gpuList.end(); jit++) {
        gpuInfo = (*jit);
        if (strncmp(gpuInfo.uuid.bytes, uuid, FM_UUID_BUFFER_SIZE) == 0) {
            // found the gpu in the excluded list
            return true;
        }
    }

    // the gou is not in the excluded list
    return false;
}

void
GlobalFabricManager::pruneNonDetectedLWSwitches(void)
{
    int disabledSwitchCount = mTopoValidator->disableSwitches();
    if ( disabledSwitchCount > 0 ) {
        std::stringstream ss;
        ss << "disabling " << disabledSwitchCount <<" LWSwitch(es) which is not detected/present on the system";
        // not logging to syslog as this LWSwitch pruning will happen in single baseboard case 
        // and full pass through virtualization case
        FM_LOG_INFO("%s", ss.str().c_str());
    }

#ifdef DEBUG
    // gather a list of disabled switches or GPUs from a conf file (manual disable)
    mpParser->parseFabricTopologyConf(DEFAULT_TOPOLOGY_CONF_FILE, mSwitchArchType);
#endif

    // modify the fabric if there are disabled switches, before configuring trunk ports
    if ( mpParser->getNumDisabledSwitches() > 0 ) {
        mpParser->modifyFabric(FMFabricParser::PRUNE_SWITCH, mSwitchArchType);
    }
}

void
GlobalFabricManager::pruneNonDetectedGpus(void)
{
    // gather a list of disabled GPUs that are not detected
    int disabledGpuCount = mTopoValidator->disableGpus(mGpuLWLinkConnMatrix);
    if ( disabledGpuCount > 0 ) {
        std::stringstream outStr;
        outStr << "disabling " << disabledGpuCount <<" GPU(s) which are not detected/present on the system";
        // not logging to syslog as this GPU pruning will happen in single baseboard case 
        // and full pass through virtualization case
        FM_LOG_INFO("%s", outStr.str().c_str());
    }

    // modify the fabric if there are disabled switches or GPUs, before configuring nodes
    if ( mpParser->getNumDisabledGpus() > 0 ) {
        mpParser->modifyFabric(FMFabricParser::PRUNE_GPU, mSwitchArchType);
    }
}

void
GlobalFabricManager::disableLWSwitchTrunkLinks(void)
{
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // For multi-node and loopout topologies, MULTI_NODE_TOPOLOGY will be set
    if ( isMultiNodeMode() )
        return;
#endif

    // Do not disable trunk links in MODS, MODS will frequently use topologies that put
    // trunk links in loopback on single baseboard systems and trunk links need to be
    // left enabled in those cases
#if !defined(LW_MODS) || defined(LW_MODS_GDM_BUILD)
    uint32 nodeId = 0; // TODO: fix this for multi-node case

    if (!mpParser->isSwtichGpioPresent()) {
        // skip if switch GPIO IDs are not specified in topology file
        return;
    }

    if( mLwswitchInfoMap.size() > 1) {
        // Do not disable trunk links for Multi-node
        FM_LOG_DEBUG("%ld node system, ignore disabling trunk LWLinks", mLwswitchInfoMap.size());
        return;
    }

    // Disable trunk links only for single base board systems
    if (!isSingleBaseboard(nodeId)) {
        return;
    }

    //
    // Skip if the trunk links are already disabled in LWSwitch to prevent IOCTL errors.
    // This can happen when FM is restarted (i.e. FM initially removed the links from LWSwitch driver).
    // Until the driver is re-loaded or the LWSwitches are re-enumerated, driver will not have
    // trunk links and any attempt to remove them will result in IOCTL failures.
    //

    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);
    FMLWSwitchInfoList &switchList = it->second;
    FMLWSwitchInfo switchInfo = switchList.front();

    // get the trunk port mask derived from topology file
    uint64 trunkLinkMask;
    mpParser->getSwitchTrunkLinkMask(nodeId, switchInfo.physicalId, trunkLinkMask);
    if (!(switchInfo.enabledLinkMask & trunkLinkMask)) {
        // Trunk links are not present in driver, no need to disable them.
        return;
    }

    FM_LOG_INFO("disabling LWSwitch trunk links for single GPU base board systems");

    // go ahead and disable trunk links
    FMIntReturn_t rc;
    rc = mpConfig->configDisableSwitchTrunkLinks(nodeId);
    if ( rc != FM_INT_ST_OK ) {
        std::stringstream ss;
        ss << "failed to disable LWSwitch trunk links to support single GPU base board systems";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // Update GFM LWSwitch link mask information accordingly. (or we should fetch it again from LocalFM)
    // Note: switchList is accessed as a reference.
    FMLWSwitchInfoList::iterator jit;
    for (jit = switchList.begin(); jit != switchList.end(); jit++) {
        FMLWSwitchInfo &switchInfo = (*jit);
        uint64 trunkLinkMask;
        mpParser->getSwitchTrunkLinkMask(nodeId, switchInfo.physicalId, trunkLinkMask);
        switchInfo.enabledLinkMask = (switchInfo.enabledLinkMask & ~trunkLinkMask);
    }
#endif
}

void
GlobalFabricManager::initializeAllLWLinks(void)
{
    int retVal = 0;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // Skip LWLink initilization if ALI training method is enabled
    if (!mDisableLwlinkAli) {
        FM_LOG_DEBUG("Using ALI based LWLink training sequence");
        GFMHelper::lwLinkGetAllNodeLinkInitStatus(mLinkTrainIntf, mpParser, mLWLinkDevRepo);

        retVal = GFMHelper::lwLinkGetAllNodeDeviceLwlinkState(mLinkTrainIntf, mpParser, mLWLinkDevRepo);
        FM_LOG_INFO("Finished checking LWLink ALI Training Status"); 

        dumpLWLinkDeviceAndInitStatus();
        return;
    }
#endif

    // Do LWLink initialization for all nodes
    retVal = GFMHelper::lwLinkInitializeAllNodes(mLinkTrainIntf, mpParser, mLWLinkDevRepo);
    GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
            "lost socket connection between global and local fabric manager instance when initializing LWLinks");
    dumpLWLinkDeviceAndInitStatus();
}

void
GlobalFabricManager::discoverAllLWLinks(void)
{
    int retVal = 0;

    // discover all the intra node connections
    retVal = GFMHelper::lwLinkDiscoverIntraNodeConnOnNodes(mpParser, mLinkTrainIntf, mLWLinkConnRepo);
    GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                       "lost socket connection between global and local fabric manager instance when discovering LWLink connections");

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // Send discovery tokens for SV10 and read SIDS for LR10 and beyond
    // both mechanisms should work for LR10
    if ( isParallelTrainingEnabled() ) {
        // read SIDS of all the inter node connections
        retVal = GFMHelper::lwLinkReadInterNodeLinkSids(mLinkTrainIntf, mpParser, mLWLinkConnRepo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
    } else {
        // discover all the inter node connections
        retVal = GFMHelper::lwLinkDiscoverInterNodeConnections(mLinkTrainIntf, mpParser, mLWLinkConnRepo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
    }

    if (isMultiNodeMode() && mSwitchArchType == LWSWITCH_ARCH_TYPE_LR10 && mpParser->isSlotIdProvided()) {
        mLWLinkConnRepo.dumpAllDiscoveredConnectionPortInfo(this, mLWLinkDevRepo);
    }

    // populate all inter node connections on the respective nodes
    retVal = GFMHelper::lwlinkAddInterNodeConnections(mLinkTrainIntf, mLWLinkConnRepo, mLWLinkDevRepo);
    GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                       "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
#endif
}

void
GlobalFabricManager::trainLWLinkConns(FMLWLinkTrainType trainTo)
{
    int retVal = 0;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)

    if ( isParallelTrainingEnabled() && trainTo == LWLINK_TRAIN_SAFE_TO_HIGH) {
        retVal = GFMHelper::lwLinkTrainInterNodeConnectionsParallel(this, mLinkTrainIntf, mLWLinkConnRepo,
                                                                    mLWLinkDevRepo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
    } else {
        retVal = GFMHelper::lwLinkTrainInterNodeConnections(this, mLinkTrainIntf, mLWLinkConnRepo,
                                                            mLWLinkDevRepo, trainTo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
	}
#endif
    
    if ( isParallelTrainingEnabled() ) {
        retVal = GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(this, mLinkTrainIntf, mLWLinkConnRepo,
                                                                    mLWLinkDevRepo, trainTo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
    } else {
        retVal = GFMHelper::lwLinkTrainIntraNodeConnections(this, mLinkTrainIntf, mLWLinkConnRepo,
                                                            mLWLinkDevRepo, trainTo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                                           "lost socket connection between global and local fabric manager instance when discovering LWLink connections");
    }

    mLWLinkConnRepo.dumpAllConnAndStateInfo(this, mLWLinkDevRepo);
}

void
GlobalFabricManager::startNonSharedFabricMode(void)
{
    int lwlinkRetrainCount = mFmLWlinkRetrainCount;

    startNonSharedInitialization();
    
    // MODS would rather ilwoke LWLink discovery and training separately in stages
    if ( mStagedInit ) {
        return;
    }

    FM_LOG_DEBUG("LWLink Retrain Count %d\n", lwlinkRetrainCount); 

    bool validationStatus = false;
    // This loop exelwtes (mFmLWlinkTrainingCount + 1) times so links can be checked
    // after calling reinitLWLinks upto mFmLWlinkTrainingCount times
    for( ;lwlinkRetrainCount >= 0; lwlinkRetrainCount-- )
    {
        FM_LOG_INFO("Initializing all links"); 
        initializeAllLWLinks();
        FM_LOG_INFO("Link initialization done"); 
        discoverAllLWLinks();
        FM_LOG_INFO("Links discovered"); 
        trainLWLinkConns(LWLINK_TRAIN_SAFE_TO_HIGH);
        FM_LOG_INFO("Links trained"); 

        GlobalFMLWLinkConnRepo failedConnections;
        if( ( ( validationStatus = validateLWLinkConns( failedConnections ) ) != true ) && (lwlinkRetrainCount > 0) )
        {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            GFMHelper::getGradingAndFomValues(this, mLinkTrainIntf, failedConnections, mLWLinkDevRepo);
            printAllFailedConnsFomAndGradingValues(failedConnections);
            reinitLWLinks( failedConnections );
#endif
        }
        else 
            break;
    }

    if( validationStatus != true )
    {
        std::ostringstream ss;
        ss << "some internode LWLink trunk (LWSwitch to LWSwitch) connections are not in Active state";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    FM_LOG_INFO("All Links LWLink trunk (LWSwitch to LWSwitch) connections trained to high speed"); 

    degradeLWLinkDevices();

    // all links are trained, populate GPU LWLink speed information
    GFMHelper::getGpuLWLinkSpeedInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mGpuLWLinkSpeedInfoMap);
 
    // configure all switch ports and routing tables on all nodes
    // all routing entries are valid in non-shared fabric mode
    configureAllFabricNodes();
}

void
GlobalFabricManager::startNonSharedInitialization(void)
{
    // Non-Shared Fabric Mode support the following configurations
    // 1. Bare Metal
    // 2. HGX-2 Multi-host
    // 3. Full pass-through (GPU & LWSwitch) multi-tenancy

    // form a map of excluded switches and their connected switches (needed to degrade peers of excluded switches)
    buildConnectedSwitchMappingForExcludedSwitches();

    if ( !mDisablePruning ) {
        // prune switches that are in the topology but not discovered by the node
        // this can happen in Multi-host/multi-tenancy mode
        pruneNonDetectedLWSwitches();
    }

    // In Multi-host, we should disable all the trunk links from switch driver
    // to prevent driver from initializing them as part of link training.
    disableLWSwitchTrunkLinks();

    // Query all the device/links from LWLinkCoreLib driver after disabling unwanted links.
    GFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);
    // configure all switch trunk ports before link training start
    // access ports would be pruned, and will be configured later,
    // after GPU and access port pruning is determined from discovered topology
    configureAllTrunkPorts();

#ifdef SIM_BUILD
    // TODO: Fix LwLink initialization in fmodel simulation
    // Links are initialized implicitly in the driver during initialization,
    // So calling all of lwLinkInitializeAllNodes will cause an error,
    // But lwLinkGetLinkInitStatusForNodes still needs to be called in order to populated the status arrays.
    GFMHelper::lwLinkGetAllNodeLinkInitStatus(mLinkTrainIntf, mpParser, mLWLinkDevRepo);
    dumpLWLinkDeviceAndInitStatus();
#endif
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void 
GlobalFabricManager::printAllFailedConnsFomAndGradingValues(GlobalFMLWLinkConnRepo &failedConnections)
{
   //iterate over all failed connections
    LWLinkInterNodeConns interConnList = failedConnections.getInterConnections();
    for( LWLinkInterNodeConns::iterator it = interConnList.begin(); it != interConnList.end(); it++ ) {
        FMLWLinkDetailedConnInfo *conn = *it;
        conn->printFomAndGradingValues(this, mLWLinkDevRepo);
    }

}
void
GlobalFabricManager::resetAndDrainLinks( GlobalFMLWLinkConnRepo &failedConnections )
{
    int retVal = 0;

    typedef std::pair<FMNodeId_t, FMGpuOrSwitchId_t> switch_t;
    typedef std::map< switch_t, FMSwitchPortMask_t > switchMasks_t;
    switchMasks_t switchMasks;

    FM_LOG_DEBUG("Entering");
    // Build a mapping of switch to the portmask that needs to be reset
    LWLinkInterNodeConns interConnList = failedConnections.getInterConnections();
    for( LWLinkInterNodeConns::iterator it = interConnList.begin(); it != interConnList.end(); it++ ) {
        FMLWLinkDetailedConnInfo *conn = *it;
        FMLWLinkEndPointInfo masterEnd = conn->getMasterEndPointInfo();
        FMLWLinkEndPointInfo slaveEnd = conn->getSlaveEndPointInfo();
        
        switchMasks_t::iterator switchMasksIt;
        FMSwitchPortMask_t tmpMask;
        // save the master end switch link mask
        tmpMask = 0;
        if( (switchMasksIt = switchMasks.find( std::make_pair( masterEnd.nodeId, masterEnd.gpuOrSwitchId)))
                                               != switchMasks.end() ) {
            tmpMask = switchMasksIt->second;
        }
        tmpMask |= 1 << masterEnd.linkIndex;
        switchMasks[ std::make_pair( masterEnd.nodeId, masterEnd.gpuOrSwitchId) ] = tmpMask;

        // save the slave end switch link mask
        tmpMask = 0;
        if( (switchMasksIt = switchMasks.find( std::make_pair( slaveEnd.nodeId, slaveEnd.gpuOrSwitchId)))
                                               != switchMasks.end() ) {
            tmpMask = switchMasksIt->second;
        }
        tmpMask |= 1 << slaveEnd.linkIndex;
        switchMasks[ std::make_pair( slaveEnd.nodeId, slaveEnd.gpuOrSwitchId) ] = tmpMask;
    }

    // For each switch call "reset and drain"
    for( switchMasks_t::iterator it = switchMasks.begin(); it != switchMasks.end(); it++ ) {
        FMNodeId_t nodeId = it->first.first;
        FMGpuOrSwitchId_t gpuOrSwitchId = it->first.second;
        FMSwitchPortMask_t tmpMask = it->second;

        FMLWLinkDevInfo devInfo;
        mLWLinkDevRepo.getDeviceInfo( nodeId, gpuOrSwitchId, devInfo);

        FMPhysicalId_t phyId=0;
        getLWSwitchPhysicalId( nodeId, devInfo.getDevicePCIInfo(), phyId);

        retVal = GFMHelper::lwLinkSendResetSwitchLinks( nodeId, phyId, tmpMask, mLinkTrainIntf);
        if( retVal != 0) 
        {
            std::ostringstream ss;
            ss << "request to reset failing LWSwitch LWLinks failed";
            ss << " for NodeId " << nodeId << " physicalId " << phyId;
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
}

void
GlobalFabricManager::reinitLWLinks(GlobalFMLWLinkConnRepo &failedConnections)
{
    // reset and drain links
    resetAndDrainLinks(failedConnections);
    // clear all internode links. these will be discovered again. If all is well the driver 
    // will not retry any links that are already in high speed. 
    // Thisis to make sure that if retraining one bad link makes a good link fail we need to catch this case
    mLWLinkConnRepo.clearConns();
}
#endif

bool
GlobalFabricManager::validateLWLinkConns( GlobalFMLWLinkConnRepo &failedConnections )
{
    // build GPU link status mapping based on detected LWLink connections.
    // the corresponding GPU link status will be false if the connection is not in Active state.
    // the non detected GPU will be later pruned by the below disableGpus()'s logic.
    // Do this mapping first to populate corresponding GPU's link enabled mask.
    mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
    dumpGpuConnMatrix();

    // log any LWLink initialization failures
    mTopoValidator->checkLWLinkInitStatusForAllDevices();

    if( !mEnableTopologyValidation )
        return true;

    // MODS does not care aout partition info whichis primarily what is checked in validateTopologyInfo
    //
    // only do the topology validation when it is enabled
    // do all the validation after link initialization and training 
    // so that all the failures are reported
    // Also do validation always in DGX2 systems trunk link failures should result in FM being aborted. 
    // Since degraded mode processing is disabled in DGX 2 systems, we need this check explicitly to 
    // ensure FM exits on trunk link failures.
    //
    if ( mEnableTopologyValidation || mSwitchArchType == LWSWITCH_ARCH_TYPE_SV10) {
#ifndef LW_MODS // validateTopologyInfo validates partitions which are lwrrently unsupported in MODS
        if (!mTopoValidator->validateTopologyInfo()) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            // TODO There is no partition info lwrrently created for multi-node systems, so this will always return
            // error. Just log a debug message and continue for now
            if (mLwswitchInfoMap.size() >= 2)
            {
                FM_LOG_DEBUG("detected LWSwitch/LWLink trunk connection information does not match topology");
            }
            else
#endif
            {
                std::ostringstream ss;
                ss << "detected LWSwitch/LWLink trunk connection information does not match topology";
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
#endif // LW_MODS
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
        failedConnections.clearConns();
        int numConfigConns;
        // check if all the configured trunk connections got to active.
        if( !mTopoValidator->isTrunkConnsActive( failedConnections, numConfigConns ) ) {
            if( (mTopoValidator->numActiveInternodeTrunks() + mTopoValidator->numActiveIntranodeTrunks()) 
                != numConfigConns ) {
                return false;
            } else {
                FM_LOG_DEBUG("numActiveInternodeTrunks =%d numConfigConns=%d", mTopoValidator->numActiveInternodeTrunks(), numConfigConns);
                // if connections have been miswired but all connections go to active,
                // we can get failed connections but no point in retrying.
                std::ostringstream ss;
                ss << "some internode LWLink trunk (LWSwitch to LWSwitch) connections are not in Active state";
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
#elif !defined(LW_MODS)
        // validate LWLink trunk connection states
        if (!mTopoValidator->isAllIntraNodeTrunkConnsActive()) {
            std::ostringstream ss;
            ss << "not all the intra node LWLink trunk (LWSwitch to LWSwitch) connections are in Active state";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
#endif
    }
    return true;
}

void
GlobalFabricManager::degradeLWLinkDevices()
{
    //
    // before degrading or pruning GPUs, report the LWSwitch LWLink training failed mask to localFM,
    // which will then be reported via out-of-band. Otherwise, if the degradation logic
    // train off some links, customer won't be able to identify the exact failed links
    // as failed links and trained off links will be in OFF state
    //
    // Note: out-of-band is not supported in Willow or in simulation
    // 
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // Lwrrently degrade mode is not supported for multi-node.
    FM_LOG_INFO("OOB reporting not supported for multi-node");
    return;
#else
    if ((mSwitchArchType != LWSWITCH_ARCH_TYPE_SV10) && (!mSimMode)) {
        if (FM_INT_ST_OK != sendLWSwitchTrainingFailedLinkInfoToAllNodes()) {
            std::ostringstream ss;
            ss << "request to report LWSwitch LWLink training failed information to LWSwitch driver failed";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
#endif

    if ( !mDisablePruning ) {
        // remove all the non-detected GPUs from fabric config context to
        // prevent any config attempt.
        pruneNonDetectedGpus();
    }

    //
    // Degraded mode processing is required for bare metal, pass through and shared fabric.
    // However, no degraded logic if FM is started in restart mode.
    //
    // NOTE: degraded mode processing is not done for baremetal on volta/willow due to issues in bug  2801911
    if (mSwitchArchType != LWSWITCH_ARCH_TYPE_SV10) {
        // process all lwlink and lwswitch failures detected during initialization
        // if degraded mode is not disabled
        if (!mDisableDegradedMode) {
            mpDegradedModeMgr->processFailures();
        }
    }
}

bool
GlobalFabricManager::finishNonSharedInitialization()
{
    // build GPU link status mapping based on detected LWLink connections.
    // the corresponding GPU link status will be false if the connection is not in Active state.
    // the non detected GPU will be later pruned by the below disableGpus()'s logic.
    // Do this mapping first to populate corresponding GPU's link enabled mask.
    mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
    dumpGpuConnMatrix();
    // log any LWLink initialization failures
    mTopoValidator->checkLWLinkInitStatusForAllDevices();

    //
    // only do the topology validation when it is enabled
    // do all the validation after link initialization and training 
    // so that all the failures are reported
    // Also do validation always in DGX2 systems trunk link failures should result in FM being aborted. 
    // Since degraded mode processing is disabled in DGX 2 systems, we need this check explicitly to 
    // ensure FM exits on trunk link failures.
    //
    if ( mEnableTopologyValidation || mSwitchArchType == LWSWITCH_ARCH_TYPE_SV10) {
#ifndef LW_MODS // validateTopologyInfo validates partitions which are lwrrently unsupported in MODS
        if (!mTopoValidator->validateTopologyInfo()) {
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            // TODO There is no partition info lwrrently created for multi-node systems, so this will always return
            // error. Just log a debug message and continue for now
            if (mLwswitchInfoMap.size() >= 2)
            {
                FM_LOG_DEBUG("detected LWSwitch/LWLink trunk connection information does not match topology");
            }
            else
#endif
            {
                std::ostringstream ss;
                ss << "detected LWSwitch/LWLink trunk connection information does not match topology";
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
        // validate LWLink trunk connection states
        if (!mTopoValidator->isAllIntraNodeTrunkConnsActive()) {
            std::ostringstream ss;
            ss << "not all the intra node LWLink trunk (LWSwitch to LWSwitch) connections are in Active state";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
#endif
    }

    //
    // before degrading or pruning GPUs, report the LWSwitch LWLink training failed mask to localFM,
    // which will then be reported via out-of-band. Otherwise, if the degradation logic
    // train off some links, customer won't be able to identify the exact failed links
    // as failed links and trained off links will be in OFF state
    //
    // Note: out-of-band is not supported in Willow or in simulation
    // 
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    FM_LOG_INFO("OOB reporting not supported for multi-node");
#else
    if ((mSwitchArchType != LWSWITCH_ARCH_TYPE_SV10) && (!mSimMode)) {
        if (FM_INT_ST_OK != sendLWSwitchTrainingFailedLinkInfoToAllNodes()) {
            std::ostringstream ss;
            ss << "request to report LWSwitch LWLink training failed information to LWSwitch driver failed";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }
#endif

    if ( !mDisablePruning ) {
        // remove all the non-detected GPUs from fabric config context to
        // prevent any config attempt.
        pruneNonDetectedGpus();
    }

    //
    // Degraded mode processing is required for bare metal, pass through and shared fabric.
    // However, no degraded logic if FM is started in restart mode.
    //
    // NOTE: degraded mode processing is not done for baremetal on volta/willow due to issues in bug  2801911
    if (mSwitchArchType != LWSWITCH_ARCH_TYPE_SV10) {
        // process all lwlink and lwswitch failures detected during initialization
        // if degraded mode is not disabled
        if (!mDisableDegradedMode) {
            mpDegradedModeMgr->processFailures();
        }
    }

    // all links are trained, populated GPU LWLink speed information
    GFMHelper::getGpuLWLinkSpeedInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mGpuLWLinkSpeedInfoMap);
 
    // configure all switch ports and routing tables on all nodes
    // all routing entries are valid in non-shared fabric mode
    configureAllFabricNodes();
    return true;
}

void
GlobalFabricManager::startSharedFabricMode(void)
{
    startSharedInitialization();
    
    if ( mStagedInit ) {
        return;
    }
    
    initializeAllLWLinks();
    discoverAllLWLinks();
    trainLWLinkConns(LWLINK_TRAIN_SAFE_TO_HIGH);

    finishSharedInitialization();
}

void
GlobalFabricManager::startSharedInitialization(void)
{
    //
    // In vGPU Mode, MIG enabled GPUs are allowed during FM initialization.
    //
    // In Shared LWSwitch Mode, MIG enabled GPUs are not allowed during FM
    // initialization. So, we need to make sure that all the GPU LWLinks are
    // enabled when starting FM since GPU LWLinks may be disabled due to MIG
    // mode. But we expect MIG mode to be disabled when starting FM in Shared
    // LWSwitch mode. Otherwise, we will not be able to map the GPU physical
    // ID and corresponding partitions. MIG mode can be enabled along with
    // partition activation flow.
    //
    uint32 nodeId = 0; // TODO: fix this for multi-node case
    if ((mFabricMode == FM_MODE_SHARED_LWSWITCH) && (!isAllGpuLWLinksEnabled(nodeId))) {
        std::ostringstream ss;
        ss << "all the GPU LWLinks must be enabled when starting fabric manager in shared fabric mode";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // form a map of excluded switches and their connected switches (needed to degrade peers of excluded switches)
    buildConnectedSwitchMappingForExcludedSwitches();

    // get partitions for the excluded switches. It would be needed if the config option is to disable partitions
    buildPartitionsWithExcludedSwitch();
    
    if ( !mDisablePruning ) {
        // prune switches that are in the topology but not discovered by the node
        // this can happen in Multi-host/multi-tenancy mode
        pruneNonDetectedLWSwitches();
    }

    // In Multi-host, we should disable all the trunk links from switch driver
    // to prevent driver from initializing them as part of link training.
    disableLWSwitchTrunkLinks();

    // Query all the device/links from LWLinkCoreLib driver.
    GFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);

    // configure all switch trunk ports before link training start
    configureAllTrunkPorts();
}

void
GlobalFabricManager::finishSharedInitialization(void)
{    
    std::ostringstream ss;
    int retVal = 0;

    // build GPU link status mapping based on detected LWLink connections.
    // the corresponding GPU link status will be false if the connection is not in Active state.
    // the non detected GPU will be later pruned by the below disableGpus()'s logic.
    // Do this mapping first to populate corresponding GPU's link enabled mask.
    mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
    dumpGpuConnMatrix();

    // log any LWLink initialization failures
    mTopoValidator->checkLWLinkInitStatusForAllDevices();

    // validate LWLink trunk connection states only when topology validation is enabled
    if (mEnableTopologyValidation && !mTopoValidator->isAllIntraNodeTrunkConnsActive()) {
        ss << "not all the LWLink trunk (LWSwitch to LWSwitch) connections are in Active state";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());        
    }

    //
    // before degrading or pruning GPUs, report the LWSwitch LWLink training failed mask to localFM,
    // which will then be reported via out-of-band. Otherwise, if the degradation logic
    // train off some links, customer won't be able to identify the exact failed links
    // as failed links and trained off links will be in OFF state
    //
    // Note: out-of-band is not supported in Willow
    // 
    if (mSwitchArchType != LWSWITCH_ARCH_TYPE_SV10) {
        if (FM_INT_ST_OK != sendLWSwitchTrainingFailedLinkInfoToAllNodes()) {
            ss << "request to report LWLink training failed information to LWSwitch driver failed";
            FM_LOG_ERROR("%s", ss.str().c_str());
            FM_SYSLOG_ERR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
    }

    if ( !mDisablePruning ) {
        // remove all the non-detected GPUs from fabric config context to
        // prevent any config attempt.
        pruneNonDetectedGpus();
    }

    //
    // Degraded mode processing is required for bare metal, pass through and shared fabric.
    // However, no degraded logic if FM is started in restart mode.
    //
    if ( isFabricModeRestart() == false) {
        // process all lwlink and lwswitch failures detected during initialization
        // if degraded mode is not disabled
        if (!mDisableDegradedMode) {
            mpDegradedModeMgr->processFailures();
        }
    }
 
    // all links are trained, populated GPU LWLink speed information
    GFMHelper::getGpuLWLinkSpeedInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mGpuLWLinkSpeedInfoMap);

    // create fabric partition manager object
    mGfmPartitionMgr->buildPartitionMappings();

    // configure all switch ports and routing tables on all nodes
    // all routing entries are invalid in shared lwswitch mode
    if (mFabricMode == FM_MODE_SHARED_LWSWITCH) {
        configureAllFabricNodes();
    }

    bool isDegraded = mpDegradedModeMgr->isAnyDeviceDegraded();
    GlobalFMLWLinkConnRepo lwLinkConnRepo;
    GlobalFMLWLinkDevRepo lwLinkDevRepo;

    //
    // fetch LWLink device and connections again as, if a device is degraded, it will be
    // unregistered from LWLinkCoreLib Driver and that will remove already detected 
    // and trained connections.
    //
    if (isDegraded) {
        // TODO: revisit for multi node
        FMNodeId_t nodeId = 0;
        retVal = GFMHelper::getLWLinkDeviceInfoFromNode(nodeId, mDevInfoMsgHndlr, lwLinkDevRepo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                        "lost socket connection between global and local fabric manager instance when getting LWLink devices");

        retVal = GFMHelper::lwLinkGetIntraNodeConnOnNodes(nodeId, mLinkTrainIntf, lwLinkConnRepo);
        GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                        "lost socket connection between global and local fabric manager instance when getting LWLink connections");
    }

    // In vGPU Mode, FM will be running in the host. So, all the LWSwitches and Physical
    // GPUs will be owned and managed from the host, similar to bare-metal. So, no need
    // to reset any LWSwitch links or detach any GPUs after FM initialization or during
    // GPU partition activation or deactivation calls.
    if (mFabricMode == FM_MODE_VGPU) {
        return;
    }

    // reset all LWSwitch links before detaching the GPUs. Before activating the partition,
    // GPUs will be SBRed by Hypervisor. So, both GPU and LWSwitch side LWLinks will be in INIT
    // state for training for partition activation. Also train the connections to OFF
    // before starting reset to avoid GPU side links from flagging errors.

    // Note: Not bailing out on error links power down. Later when a partition with corresponding
    // LWLinks is activated, that activation may/may not fail. Also, there may be other partitions
    // which are not using the failed LWLinks, which can be activated successfully.
    if ( isParallelTrainingEnabled() ) {
        retVal = GFMHelper::lwLinkTrainIntraNodeConnectionsParallel(this, mLinkTrainIntf, isDegraded ? lwLinkConnRepo : mLWLinkConnRepo,
                                                                     isDegraded ? lwLinkDevRepo: mLWLinkDevRepo, LWLINK_TRAIN_TO_OFF);
    } else {
        retVal = GFMHelper::lwLinkTrainIntraNodeConnections(this, mLinkTrainIntf, isDegraded ? lwLinkConnRepo : mLWLinkConnRepo,
                                                             isDegraded ? lwLinkDevRepo: mLWLinkDevRepo , LWLINK_TRAIN_TO_OFF);
    }
    GFMHelper::logErrAndThrowException(retVal, FM_LWLINK_ST_MASTER_FM_SOCKET_ERR,
                "lost socket connection between global and local fabric manager instance when training LWLink connections ");

    GFMHelper::lwLinkResetAllSwitchLinks(0, mLinkTrainIntf);
    mpConfig->configDetachAllGPUs(0);
}

/*************************************************************************************
 * A brief flow to FM restart in Shared LWSwitch or vGPU based multitenancy mode
 *
 *  1. Skip all Link training and all the GPU and LWSwitch configurations/initialization.
 *  2. HaMgr open, parse state file and load the states to PartitionMgr.
 *  3. PartitionMgr wait up to one minute for the hypervisor to notify current activated
 *     partitions via setActivatedPartition API.
 *  4. PartitionMgr set mInitDone, so that normal partition activation and deactivation
 *     can start
 *************************************************************************************/
void
GlobalFabricManager::restartSharedFabricMode(void)
{
    uint32 nodeId = 0; // TODO: fix this for multi-node case

    FM_LOG_DEBUG("FM restarted in %d mode", mFabricMode);

    if (mFabricMode == FM_MODE_SHARED_LWSWITCH) {
        //
        // in shared vm restart mode, we are not expecting any GPUs to be attached to RM
        // so, check the probed GPU count and abort if we got some GPUs attached. Also, in 
        // this mode, we are not detaching any GPUs as part of initialization. So the 
        // GPUs will be in attached state if we continue.
        //
        if (mGpuInfoMap.size()) {
            FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
            if (gpuList.size() != 0) {
                std::ostringstream ss;
                ss << "all the GPUs must be detached from LWPU driver when starting fabric manager in shared lwswitch resiliency mode";
                FM_LOG_ERROR("%s", ss.str().c_str());
                FM_SYSLOG_ERR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
            }
        }
    }

    // Query all the device/links from LWLinkCoreLib driver after restart
    GFMHelper::getLWLinkDeviceInfoFromAllNodes(mpParser, mDevInfoMsgHndlr, mLWLinkDevRepo);

    if ( !mStagedInit ) {
        // initialize all the links, discover connectiLons, but do NOT train them.
        initializeAllLWLinks();
        discoverAllLWLinks();
    }

    if (mFabricMode == FM_MODE_VGPU) {
        // Train LWLinks to make sure that all connections are trained and active.
        trainLWLinkConns(LWLINK_TRAIN_SAFE_TO_HIGH);

        //
        // Build GPU link status mapping based on detected LWLink connections.
        // Do this mapping first to populate corresponding GPU's link enabled mask.
        //
        mTopoValidator->mapGpuIndexByLWLinkConns(mGpuLWLinkConnMatrix);
        dumpGpuConnMatrix();
    }
}

void
GlobalFabricManager::finishInitialization(void)
{
    if (mFabricMode == FM_MODE_BAREMETAL) {
        finishNonSharedInitialization();
    } else {
        if ( !isFabricModeRestart() ) {
            finishSharedInitialization();
        }
    }
    
    finishGlobalFMInitialization();
}

uint32_t
GlobalFabricManager::getControlMessageRequestId(uint32_t fabricNodeId)
{
    std::map <uint32_t, FMFabricNode *>::iterator it;

    it = mvFabricNodes.find( fabricNodeId );
    if ( it == mvFabricNodes.end() )
    {
        FM_LOG_DEBUG("Invalid fabric " NODE_ID_LOG_STR " %d", fabricNodeId);
        return 0;
    }

    FMFabricNode *pFabricNode = it->second;
    return pFabricNode->getControlMessageRequestId();
}

void
GlobalFabricManager::setNodeConfigError(uint32_t nodeId)
{
    std::map <uint32_t, FMFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        FMFabricNode *pNode = it->second;
        pNode->setConfigError();
    }
}

void
GlobalFabricManager::clearNodeConfigError(uint32_t nodeId)
{
    std::map <uint32_t, FMFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        FMFabricNode *pNode = it->second;
        pNode->clearConfigError();
    }
}

bool
GlobalFabricManager::isNodeConfigErrorOclwred(uint32_t nodeId)
{
    std::map <uint32_t, FMFabricNode *>::iterator it;
    it = mvFabricNodes.find( nodeId );

    if ( it != mvFabricNodes.end() )
    {
        FMFabricNode *pNode = it->second;
        return pNode->isConfigError();
    }
    return false;
}

void
GlobalFabricManager::refreshGpuLWLinkMaskInfo(uint32_t nodeId, char *gpuUuid)
{
    // first re-fetch the information from specified node.
    FMGpuInfoMap tempGpuInfoMap;
    FMExcludedGpuInfoMap tempExcludedGpuInfoMap;
    GFMHelper::getGpuDeviceInfoFromNode(nodeId, mDevInfoMsgHndlr, tempGpuInfoMap, tempExcludedGpuInfoMap);

    // get the new mask from newly retrieved gpu information
    FMGpuInfoList tempGpuList = tempGpuInfoMap[nodeId];
    FMGpuInfoList::iterator it;
    FMGpuInfo_t newGpuInfo = {0};
    bool bFound = false;
    for (it = tempGpuList.begin(); it != tempGpuList.end(); it++) {
        newGpuInfo = (*it);
        if (strncmp(newGpuInfo.uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE) == 0) {
            bFound = true;
            break;
        }
    }

    // check whether we found the specified GPU
    if (bFound == false) {
        // not found the requested gpu in new list, return
        return;
    }

    // update our globalFM GPU map with new link mask
    FMGpuInfoList& gpuList = mGpuInfoMap[nodeId];
    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        FMGpuInfo_t tempInfo = (*it);
        if (strncmp(tempInfo.uuid.bytes, gpuUuid, FM_UUID_BUFFER_SIZE) == 0) {
            // found the desired one, remove and re-add it with updated information
            gpuList.erase(it);
            tempInfo.enabledLinkMask = newGpuInfo.enabledLinkMask;
            tempInfo.discoveredLinkMask = newGpuInfo.discoveredLinkMask;
            gpuList.push_back(tempInfo);
            break;
        }
    }
}

bool
GlobalFabricManager::getGpuInfo(char uuid[], FMGpuInfo_t &gpuInfo)
{
    FMGpuInfoMap::iterator it;

    for ( it = mGpuInfoMap.begin(); it != mGpuInfoMap.end(); it++ ) {
        FMGpuInfoList gpuList = it->second;
        FMGpuInfoList::iterator jit;

        for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ ) {
            FMGpuInfo_t gInfo = (*jit);
            if (strncmp(uuid, gInfo.uuid.bytes, FM_UUID_BUFFER_SIZE) == 0)
            {
                // found the GPU, return the info
                gpuInfo = gInfo;
                return true;
            }
        }
    }

    return false;
}

bool
GlobalFabricManager::getGpuInfo(uint32_t nodeId, FMPciInfo_t &pciInfo, FMGpuInfo_t &gpuInfo)
{
    FMGpuInfoMap::iterator it = mGpuInfoMap.find(nodeId);
    if ( it == mGpuInfoMap.end() )
    {
        return false;
    }

    FMGpuInfoList gpuList = it->second;
    FMGpuInfoList::iterator jit;

    for ( jit = gpuList.begin(); jit != gpuList.end(); jit++ )
    {
        FMGpuInfo_t gInfo = (*jit);
        if ((gpuInfo.pciInfo.domain == pciInfo.domain) &&
            (gpuInfo.pciInfo.bus == pciInfo.bus) &&
            (gpuInfo.pciInfo.device == pciInfo.device) &&
            (gpuInfo.pciInfo.function == pciInfo.function))
        {
            // found the GPU, return the info
            gpuInfo = gInfo;
            return true;
        }
    }

    return false;
}

bool
GlobalFabricManager::getGpuInfo(uint32_t nodeId, uint32_t physicalId, FMGpuInfo_t &gpuInfo)
{
    char uuid[FM_UUID_BUFFER_SIZE];
    if ( getGpuUuid(nodeId, physicalId, uuid) == false )
    {
        return false;
    }

    return getGpuInfo(uuid, gpuInfo);
}

uint64_t
GlobalFabricManager::getGpuEnabledLinkMask(char uuid[])
{
    FMGpuInfo_t gpuInfo;

    if ( getGpuInfo(uuid, gpuInfo) == false )
    {
        // the GPU is not found
        return 0;
    }

    return gpuInfo.enabledLinkMask;
}

bool
GlobalFabricManager::isParallelTrainingEnabled()
{
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    if ((mSwitchArchType == LWSWITCH_ARCH_TYPE_LR10) ||
        (mSwitchArchType == LWSWITCH_ARCH_TYPE_LS10)) {
#else
    if (mSwitchArchType == LWSWITCH_ARCH_TYPE_LR10) {
#endif
        // parallel link trianing is supported
        FM_LOG_DEBUG("Parallel Training is Enabled");
        return true;
    }

    FM_LOG_DEBUG("Parallel Training is Disabled");
    // by default return false
    return false;
}

lwSwitchArchType
GlobalFabricManager::getSwitchArchType()
{
    return mSwitchArchType;
}

void
GlobalFabricManager::createFmApiSocketInterafces()
{

    // first create our FMLib API interface socket.
    mGlobalFmApiHandler = new GlobalFmApiHandler(this);

    if (mFmLibCmdSockPath == NULL || !(strnlen(mFmLibCmdSockPath, FM_CONFIG_MAX_STRING_ITEM_LEN))) { 
        mGlobalFMLibCmdServer = new GlobalFMLibCmdServer( this, mGlobalFmApiHandler,
                                                          mFmLibPortNumber, mFmLibCmdBindInterface, true );
    }
    else {
        mGlobalFMLibCmdServer = new GlobalFMLibCmdServer( this, mGlobalFmApiHandler,
                                                          mFmLibPortNumber, mFmLibCmdSockPath, false );
    }

    if (0 != mGlobalFMLibCmdServer->Start()) {
        FM_LOG_ERROR("failed to create and initialize socket connection for fabric manager API interface");
        throw std::runtime_error("failed to create and initialize socket connection for fabric manager API interface");
    }

    /* Wait for the notification that the server has started running. 
       Return error if the server can't be started */
    if (0 != mGlobalFMLibCmdServer->WaitToStart()) {
        FM_LOG_ERROR("failed to get socket connection for fabric manager API interface to ready state");
        throw std::runtime_error("failed to get socket connection for fabric manager API interface to ready state");
    }

    // now create our internal FM API interface socket.

    //
    // create internal API interface socket for bare metal or vgpu mode only as it is mainly for handling GPU reset,
    // which is not applicable in shared fabric mode
    //
    mGlobalFMInternalCmdServer = NULL;
    if (mFabricMode != FM_MODE_SHARED_LWSWITCH) {
        mGlobalFMInternalCmdServer = new GlobalFMInternalCmdServer( this, mGlobalFmApiHandler);

        if (0 != mGlobalFMInternalCmdServer->Start()) {
            FM_LOG_ERROR("failed to create and initialize socket connection for fabric manager internal API interface");
            throw std::runtime_error("failed to create and initialize socket connection for fabric manager internal API interface");
        }

        /* Wait for the notification that the server has started running.
           Return error if the server can't be started */
        if (0 != mGlobalFMInternalCmdServer->WaitToStart()) {
            FM_LOG_ERROR("failed to get socket connection for fabric manager internal API interface to ready state");
            throw std::runtime_error("failed to get socket connection for fabric manager internal API interface to ready state");
        }
    }
}

bool
GlobalFabricManager::isSingleBaseboard(uint32_t nodeId)
{
    return (GFMHelper::getNumBaseboard(nodeId, mLwswitchInfoMap, mExcludedLwswitchInfoMap) == 1);
}

bool
GlobalFabricManager::isNodeDegraded(uint32_t nodeId)
{
    return (mDegradedNodes.find(nodeId) != mDegradedNodes.end()) ? true : false;
}

FMIntReturn_t
GlobalFabricManager::sendLWSwitchTrainingFailedLinkInfoToAllNodes()
{
    FMIntReturn_t retVal = FM_INT_ST_OK;

    // iterate through all the fabric node objects
    std::map <uint32_t, FMFabricNode*>::iterator it;
    for (it = mvFabricNodes.begin(); it != mvFabricNodes.end(); it++) {
        uint32_t nodeId = it->first;
        //
        // sent training failed link information for each node
        // on failure, continue to next node, but return over-all status as failure
        //
        retVal = sendLWSwitchTrainingFailedLinkInfoToNode(nodeId);
    }
    return retVal;
}

FMIntReturn_t
GlobalFabricManager::sendLWSwitchTrainingFailedLinkInfoToNode(uint32_t nodeId)
{
    FMIntReturn_t retVal = FM_INT_ST_OK;

    FMLWSwitchInfoMap::iterator it = mLwswitchInfoMap.find(nodeId);
    if (it == mLwswitchInfoMap.end()) {
        // no entry for the specified node
        return FM_INT_ST_OK;
    }

    //
    // for each detected switch, get the non-active links. If the link is used and supposed
    // to train, treat it as failed link and compute the mask accordingly and report it.
    //

    FMLWSwitchInfoList detectedSwitchList = it->second;
    FMLWSwitchInfoList::iterator jit;
    for (jit = detectedSwitchList.begin(); jit != detectedSwitchList.end(); jit++) {
        FMLWSwitchInfo detectedSwitchInfo = (*jit);

        // first get the corresponding LWLink device
        FMLWLinkDevInfo switchLwLinkDevInfo;
        if (mLWLinkDevRepo.getDeviceInfo(nodeId, detectedSwitchInfo.pciInfo , switchLwLinkDevInfo) == false) {
            // this is not expected, treat as error
            std::stringstream outStr;
            outStr << "unable to find LWSwitch pci bus id:" << detectedSwitchInfo.pciInfo.busId << " physical id:"
                   << detectedSwitchInfo.physicalId << " information in LWLink driver context";
            FM_LOG_ERROR("%s", outStr.str().c_str());
            FM_SYSLOG_ERR("%s", outStr.str().c_str());
            return FM_INT_ST_GENERIC_ERROR;            
        }

        // found the switch in our LWLink driver context. get it failed LWLinks and report the failed link mask
        retVal = sendSwitchTrainingFailedLinksInfo(nodeId, detectedSwitchInfo, switchLwLinkDevInfo);
        if (FM_INT_ST_OK != retVal) {
            // error already logged. no need to continue for this node
            return retVal;
        }
    }

    // finished updating lwlink training failed for the current node
    return retVal;
}

FMIntReturn_t
GlobalFabricManager::sendSwitchTrainingFailedLinksInfo(uint32_t nodeId, FMLWSwitchInfo &detectedSwitchInfo,
                                                        FMLWLinkDevInfo &switchLwLinkDevInfo)
{
    uint64 trainingAttemptedLinkMask0 = detectedSwitchInfo.enabledLinkMask;
    uint64 trainingFailedLinkMask0 = 0;
    std::list<uint32> nonActiveLinks;
    std::list<uint32> missingConnLinks;
    std::list<uint32>::iterator it;

    switchLwLinkDevInfo.getNonActiveLinksIndex(nonActiveLinks, missingConnLinks);
    for (it = nonActiveLinks.begin(); it != nonActiveLinks.end(); it++) {
        uint32_t linkIndex = (*it);
        //
        // found a failed link, see whether it is supposed to be connected
        //
        // Note: We can't directly do a look-up in topology to check whether the switch port is used. That will require a
        // reverse comparison (i,e compare the switch id:port against both localEndpoint and farEndpoint of the connection
        // in topology file). Such a reverse comparison could yield false positive as the switch physical ID 
        // can match with GPU physical IDs (and link index) as well and get matched with some access connections 
        // for a real trunk connection or for an unused switch id:port pair.
        //
        accessPort *accessPortInfo = NULL;
        trunkPort *trunkPortInfo = NULL;
        accessPortInfo = mpParser->getAccessPortInfoFromCopy(nodeId, detectedSwitchInfo.physicalId, linkIndex);
        if (accessPortInfo != NULL) {
            // this link was supposed to be active, report it as failed
            trainingFailedLinkMask0 |= BIT64(linkIndex);
        } else {
            // no access port, look for a trunk port
            trunkPortInfo = mpParser->getTrunkPortInfoFromCopy(nodeId, detectedSwitchInfo.physicalId, linkIndex);
            if (trunkPortInfo != NULL) {
                // this link was supposed to be active, report it as failed
                trainingFailedLinkMask0 |= BIT64(linkIndex);
            }
        }

        //
        // if we not found the port in Access or Trunk list, then this port is not used.
        // so clear the corresponding bit in attempted training bit mask
        //
        if ((accessPortInfo == NULL) && (trunkPortInfo == NULL)) {
            trainingAttemptedLinkMask0 &= ~BIT64(linkIndex);
        }
    }

    // send the computed link mask to corresponding localFM
    int tempRet = GFMHelper::lwLinkSendSwitchTrainingFailedLinkInfo(nodeId, detectedSwitchInfo.physicalId,
                                                                    trainingAttemptedLinkMask0, trainingFailedLinkMask0,
                                                                    mLinkTrainIntf);
    if (tempRet) {
        FM_LOG_ERROR("failed to set LWSwitch link training failed information for " NODE_ID_LOG_STR " %d LWSwitch physical id %d ", nodeId, detectedSwitchInfo.physicalId );
        return FM_INT_ST_GENERIC_ERROR;
    }

    return FM_INT_ST_OK;
}

bool
GlobalFabricManager::isAllGpuLWLinksEnabled(uint32_t nodeId)
{
    FMGpuInfoList gpuList = mGpuInfoMap[nodeId];
    FMGpuInfoList::iterator it;

    for (it = gpuList.begin(); it != gpuList.end(); it++) {
        FMGpuInfo_t tempGpuInfo = (*it);
        if (tempGpuInfo.discoveredLinkMask != tempGpuInfo.enabledLinkMask) {
            // not all the links are enabled
            return false;
        }
    }

    // not found any GPU or all links are enabled
    return true;
}
void
GlobalFabricManager::doSingleNodeGlobalFMPreInit(GlobalFmArgs_t *pGfmArgs)
{
    FMIntReturn_t rc;

    std::stringstream topoFileName;
    topoFileName << pGfmArgs->topologyFilePath;
    //
    // for single node systems, first open control connection and get LWSwitch architecture type
    // node Id is hardcoded to zero for single node
    //
    createFabricNode(0, pGfmArgs->fmBindInterfaceIp, pGfmArgs->domainSocketPath);

    // fabric node objects are created based on mode, wait for the FabricNode's control connections to establish
    if (!waitForAllFabricNodeCtrlConns()) {
        FM_LOG_ERROR("failed to establish control connection with all the local fabric managers");
        FM_SYSLOG_ERR("failed to establish control connection with all the local fabric managers");
        throw std::runtime_error("failed to establish control connection with all the local fabric managers");
    }

    //
    // control connections are ready, send global info config to the node before
    // querying anything from the nodes.
    //
    sendGlobalConfigToAllFabricNodes();

    //
    // technically single node don't have to validate GFM and LFM versions each other. However when
    // we run GFM and LFM as separate process and if somebody purposefully run different
    // lw-fabricmanager binary, then this version check will catch that.
    //
    checkFabricNodeVersions();
    FM_LOG_DEBUG("checkFabricNodeVersions Done");

    // for single node systems, get the LWSwitch information and open/process the topology file accordingly
    FMLWSwitchInfoMap tempLwswitchInfoMap;
    FMExcludedLWSwitchInfoMap tempExcludedLwswitchInfoMap;
    int retVal = GFMHelper::getLWSwitchDeviceInfoFromNode(0, mDevInfoMsgHndlr,
                                                          tempLwswitchInfoMap,
                                                          tempExcludedLwswitchInfoMap);
    if (retVal) {
        FM_LOG_ERROR("failed to get LWSwitch count and architecture type information");
        throw std::runtime_error("failed to get LWSwitch count and architecture type information");
    }

    // check whether the node has LWSwitches
    FMLWSwitchInfoMap::iterator it = tempLwswitchInfoMap.find(0);
    if (it == tempLwswitchInfoMap.end()) {
        FM_LOG_ERROR("No LWSwitches detected, aborting fabric manager");
        FM_SYSLOG_ERR("No LWSwitches detected, aborting fabric manager");
        throw std::runtime_error("No LWSwitches detected, aborting fabric manager");
    }

    // check whether the node's switch list is empty as well.
    FMLWSwitchInfoList &switchList = it->second;
    if (switchList.empty()) {
        FM_LOG_ERROR("No LWSwitches detected, aborting fabric manager");
        FM_SYSLOG_ERR("No LWSwitches detected, aborting fabric manager");
        throw std::runtime_error("No LWSwitches detected, aborting fabric manager");
    }

    // found some switch, check its architecture type and open topology file accordingly
    FMLWSwitchInfo switchInfo = switchList.front();
    //
    // Note: Assuming we will not have mixed switch arch types. It will be same for all
    // nodes and within a node.
    //
    mSwitchArchType = GFMHelper::driverToLwSwitchArchType(switchInfo.archType);

    // vGPU Mode is supported from LR10/GA100 and newer LWSwitch/GPU based systems
    if ((mFabricMode == FM_MODE_VGPU) && (switchInfo.archType < LWSWITCH_ARCH_TYPE_LR10)) {
        FM_LOG_ERROR("vGPU based multitenancy mode is not supported on detected LWSwitch or GPU architecture generation");
        FM_SYSLOG_ERR("vGPU based multitenancy mode is not supported on detected LWSwitch or GPU architecture generation");
        throw std::runtime_error("vGPU based multitenancy mode is not supported on detected LWSwitch or GPU architecture generation");
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // ALI based LWLink training is supported from LS10/GH100 and newer LWSwitch/GPU based systems
    if ((!mDisableLwlinkAli) && (switchInfo.archType <= LWSWITCH_ARCH_TYPE_LR10)) {
        FM_LOG_WARNING("ALI based LWLink training is not supported on detected LWSwitch or GPU architecture generation");
        mDisableLwlinkAli=true;
    }
#endif

    //
    // MODS provide absolute topology file in the config option. So, this topology autodetection 
    // is required only for Non-MODS builds
    //
#if !defined(LW_MODS) // || defined(LW_MODS_GDM_BUILD)
    if (mSwitchArchType == LWSWITCH_ARCH_TYPE_SV10) {
        // Willow based system, use DGX-2/HGX-2 topology file
        topoFileName << "/" << DGX2_HGX2_TOPOLOGY_FILENAME;

    } else if (mSwitchArchType == LWSWITCH_ARCH_TYPE_LR10) {
        // LimeRock based system, use DGX-Next/HGX-Next topology file
        topoFileName << "/" << DGXA100_HGXA100_TOPOLOGY_FILENAME;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    } else if (mSwitchArchType == LWSWITCH_ARCH_TYPE_LS10) {
        topoFileName << "/" << DGXH100_HGXH100_TOPOLOGY_FILENAME;
#endif
    } else {
        FM_LOG_ERROR("LWSwitch architecture type is not supported, aborting fabric manager");
        FM_SYSLOG_ERR("LWSwitch architecture type is not supported, aborting fabric manager");
        throw std::runtime_error("LWSwitch architecture type is not supported, aborting fabric manager");
    }
#endif

    // open corresponding topology file
    rc = parseNodeFabricTopology(topoFileName.str().c_str());

    // make sure the topology is parsed successfully
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("failed to open/parse fabric topology file information, aborting fabric manager");
        FM_SYSLOG_ERR("failed to open/parse fabric topology file information, aborting fabric manager");
        throw std::runtime_error("failed to parse fabric topology file information, aborting fabric manager");
    }
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
GlobalFabricManager::doMultiNodeGlobalFMPreInit(GlobalFmArgs_t *pGfmArgs)
{
    FMIntReturn_t rc;

    std::stringstream topoFileName;
    topoFileName << pGfmArgs->topologyFilePath;

    FM_LOG_DEBUG("Multi node topology =%s", mMultiNodeTopology);

    // domain socket is not supported in multi-node
    if ((pGfmArgs->domainSocketPath != NULL) && ( strnlen(pGfmArgs->domainSocketPath, FM_CONFIG_MAX_STRING_ITEM_LEN) != 0)) {
        FM_LOG_ERROR("Unix domain socket is not supported for multi-node configuration");
        FM_SYSLOG_ERR("Unix domain socket is not supported for multi-node configuration");
        throw std::runtime_error("Unix domain socket is not supported for multi-node configuration");
    }

    //
    // MODS provide absolute topology file in the config option. So, this directory + filename
    // concatenation is not required for MODS
    //
#if !defined(LW_MODS) //|| defined(LW_MODS_GDM_BUILD)
    topoFileName << "/" << mMultiNodeTopology;
#endif

    rc = parseMultiNodeFabricTopology(topoFileName.str().c_str());
    if ( rc != FM_INT_ST_OK ) {
        FM_LOG_ERROR("failed to open/parse fabric topology file information, aborting fabric manager");
        FM_SYSLOG_ERR("failed to open/parse fabric topology file information, aborting fabric manager");
        throw std::runtime_error("failed to open/parse fabric topology file information, aborting fabric manager");
    }

    // default arch type to SV10
    mSwitchArchType = LWSWITCH_ARCH_TYPE_SV10;

    //
    // Update arch type if the information is provided in topology file. Old topology files will not have arch defined
    // and those will fall back to default arch type, which is SV10
    //
    if( mpParser->getFabricArch() != LWSWITCH_ARCH_TYPE_ILWALID )
    {
        mSwitchArchType = mpParser->getFabricArch();
        FM_LOG_INFO( "read LWSwitch architecture from topology file" );
    }

    const char *archName=nullptr;
    if( GFMHelper::getArchNameForArchType( mSwitchArchType, &archName ) == false )
    {
        const char *errStr = "unsupported LWSwitch architecture type is specified in topology file, aborting fabric manager";
        FM_LOG_ERROR( "%s", errStr );
        FM_SYSLOG_ERR( "%s", errStr );
        throw std::runtime_error( errStr );
    }
    FM_LOG_DEBUG( "Arch type=%s", archName );

    std::map<uint32_t, std::string> nodeToIpAddrMap;
    // form node to IP address mapping
    readFabricNodeConfigFile(mFabricNodeConfigFile, nodeToIpAddrMap);
    mpParser->updateFabricNodeAddressInfo(nodeToIpAddrMap, mDegradedNodes);

    // create all the fabric node objects, to establish control connection to Local Fabric Managers
    createFabricNodes();
    
    // wait for the FabricNode's control connections to establish
    if (!waitForAllFabricNodeCtrlConns()) {
        FM_LOG_ERROR("failed to establish control connection with all the local fabric managers");
        FM_SYSLOG_ERR("failed to establish control connection with all the local fabric managers");
        throw std::runtime_error("failed to establish control connection with all the local fabric managers");
    }

    //
    // control connections are ready, send global info config to the node before
    // querying anything from the nodes.
    //
    sendGlobalConfigToAllFabricNodes();

    // all LFM to GFM connections are created. Check if all FM versions are same before proceeding
    checkFabricNodeVersions();
    FM_LOG_DEBUG("checkFabricNodeVersions Done");
}

#endif
