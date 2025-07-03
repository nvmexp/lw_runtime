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

#ifndef GlobalFabricManager_H
#define GlobalFabricManager_H
 
#include <queue>
#include <unistd.h>
#include "workqueue.h"
#include "fabricmanager.pb.h"
#include <g_lwconfig.h>
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include "fm_config_options.h"
#include "timelib.h"
#include "lwmisc.h"
#include "GlobalFmFabricConfig.h"
#include "FMLwcmClient.h"
#include "GlobalFMLWLinkIntf.h"
#include "FMLWLinkTypes.h"
#include "GlobalFMLibCmdServer.h"
#include "GlobalFMLWLinkConnRepo.h"
#include "FMDevInfoMsgHndlr.h"
#include "GlobalFmFabricNode.h"
#include "GlobalFmHaMgr.h"
#include "GlobalFMInternalCmdServer.h"
#include "GlobalFmMulticastMgr.h"

/*****************************************************************************/
/*  Global Fabric Manager is client to Local Fabric Managers for topology    */      
/*  discovery and configuration, heartbeat, and query relay to other Local   */
/*  Fabric Managers.                                                         */
/*****************************************************************************/
 
 
typedef struct {
    unsigned int fabricMode;
    bool fabricModeRestart;
    bool stagedInit;
    unsigned short fmStartingTcpPort;
    char *fmBindInterfaceIp;
    char *fmLibCmdBindInterface;
    char *fmLibCmdSockPath;
    unsigned short fmLibPortNumber;
    char *domainSocketPath;
    char *stateFileName;
    bool continueWithFailures;
    unsigned int accessLinkFailureMode;
    unsigned int trunkLinkFailureMode;
    unsigned int lwswitchFailureMode;
    bool enableTopologyValidation;
    char *fabricPartitionFileName;
    char *topologyFilePath;
    bool disableDegradedMode;
    bool disablePruning;
    int gfmWaitTimeout;
	bool simMode;
    char *fabricNodeConfigFile;
    char *multiNodeTopology;
    int fmLWlinkRetrainCount;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool disableLwlinkAli;
#endif
} GlobalFmArgs_t;
 
class GlobalFMLWLinkConnRepo;
class FMFabricConfig;
class FMFabricParser;
class GlobalFabricManager;
class GlobalFmErrorStatusMsgHndlr;
class FMTopologyValidator;
class GlobalFMCommandServer;
class GlobalFMErrorHndlr;
class GlobalFMFabricPartitionMgr;
class GlobalFMLibCmdServer;
class GlobalFmHaMgr;
class GlobalFmDegradedModeMgr;
class GlobalFMInternalCmdServer;
class GlobalFmApiHandler;

typedef struct {
    uint32_t gpuPhyIndex;
    uint32_t gpuEnumIndex;
    FMUuid_t uuid;
    bool linkConnStatus[MAX_LWLINKS_PER_GPU];
} FMGpuLWLinkConnMatrix;
 
typedef std::list<FMGpuLWLinkConnMatrix> FMGpuLWLinkConnMatrixList;
typedef std::map<uint32, FMGpuLWLinkConnMatrixList>  FMGpuLWLinkConnMatrixMap;

typedef std::map<SwitchKeyType, FMSwitchPortMask_t> ConnectedSwitchesInfoMap;
typedef std::set <PartitionKeyType> PartitionSet;
 
class GlobalFabricManager: public FMConnInterface
{
    friend class GlobalFMCommandServer;
    friend class FMTopologyValidator;
    friend class GlobalFMErrorHndlr;
    friend class FMLWLinkDevInfo;
    friend class GlobalFMFabricPartitionMgr;
    friend class GlobalFmDegradedModeMgr;
    friend class GlobalFmApiHandler;
    friend class FMFabricNode;

public:
    GlobalFabricManager(GlobalFmArgs_t *pGfmArgs);
    ~GlobalFabricManager();
 
    int ProcessMessage(uint32 nodeId, lwswitch::fmMessage * pFmMessage, bool &isResponse);
    void OnFabricNodeConnect(uint32 nodeId);
    void OnFabricNodeDisconnect(uint32 nodeId);
 
    // implementation of FMConnInterface pure virtual functions
    virtual FMIntReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );
    virtual FMIntReturn_t SendMessageToLfm( uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage,
                                            bool trackReq );
    virtual FMIntReturn_t SendMessageToLfmSync(uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage,
                                               lwswitch::fmMessage **pResponse, uint32_t timeoutSec);
    virtual uint32_t getControlMessageRequestId(uint32_t fabricNodeId);
 
    void queueErrorWorkerRequest(uint32 nodeId, lwswitch::fmMessage *errorMsg);
    bool getGpuPhysicalIndex(uint32 nodeId, uint32_t enumIndex, uint32_t &physicalIdx);
    bool getGpuEnumIndex(uint32 nodeId, uint32_t physicalIdx, uint32_t &enumIndex);
    bool getGpuPciBdf(uint32 nodeId, uint32_t enumIndex, FMPciInfo_t &pciInfo);
    bool getGpuPciBdf(uint32 nodeId, char uuid[], FMPciInfo_t &pciInfo);
    bool getGpuUuid(uint32 nodeId, uint32 physicalId, char uuid[]);
    bool getGpuNumActiveLWLinks(uint32 nodeId, FMPciInfo_t pciInfo, uint32_t &numActiveLinks);
    bool getGpuLinkSpeedInfo(uint32 nodeId, char uuid[], FMLWLinkSpeedInfo &linkSpeedInfo);
    bool getGpuDiscoveredLinkMask(uint32 nodeId, uint32_t enumIndex, uint32_t &discoveredLinkMask);
    bool getLWSwitchLWLinkDriverId(uint32 nodeId, uint32 physicalId, uint64 &lwLinkSwitchId);
    bool getGpuLWLinkDriverId(uint32 nodeId, uint32_t physicalId, uint64_t &lwLinkGpuId);
    bool getGpuLWLinkDriverId(uint32 nodeId, char uuid[], uint64_t &lwLinkGpuId);
    bool getLWSwitchPciBdf(uint32 nodeId, uint32_t physicalId, FMPciInfo_t &pciInfo);
    bool getLWSwitchPciBdf(uint32 nodeId, char uuid[], FMPciInfo_t &pciInfo);
    bool getExcludedLWSwitchPciBdf(uint32 nodeId, uint32_t physicalId, FMPciInfo_t &pciInfo);
    bool getGpuPhysicalId(uint32 nodeId, char uuid[], uint32_t &physicalId);

    void refreshGpuLWLinkMaskInfo(uint32_t nodeId, char *gpuUuid);
    bool getGpuInfo(char uuid[], FMGpuInfo_t &gpuInfo);
    bool getGpuInfo(uint32_t nodeId, FMPciInfo_t &pciInfo, FMGpuInfo_t &gpuInfo);
    bool getGpuInfo(uint32_t nodeId, uint32_t physicalId, FMGpuInfo_t &gpuInfo);

    uint64_t getGpuEnabledLinkMask(char uuid[]);
    bool getLWSwitchInfo(uint32 nodeId, uint32 physicalId, FMLWSwitchInfo &switchInfo);
    bool getLWSwitchPhysicalId(uint32 nodeId, lwlink_pci_dev_info pciDevInfo, uint32 &physicalId);
 
    unsigned short getStartingPort( ) { return mStartingPort; }
    unsigned int getFabricMode( ) { return mFabricMode; }
    bool isFabricModeRestart() { return mFabricModeRestart; }
    bool shouldContinueWithFailure( ) { return mContinueWithFailure; }
 
    const FMGpuInfoMap& getGpuInfoMap() const { return mGpuInfoMap; }
    const FMExcludedGpuInfoMap& getExcludedGpuInfoMap() const { return mExcludedGpuInfoMap; }    
    const FMLWSwitchInfoMap& getLwSwitchInfoMap() const { return mLwswitchInfoMap; }
    const FMExcludedLWSwitchInfoMap& getExcludedLwswitchInfoMap() const { return mExcludedLwswitchInfoMap; }
    bool isLwswitchExcluded( uint32 nodeId, uint32_t physicalId );
    bool isGpuExcluded( uint32 nodeId, char uuid[] );
    void buildConnectedSwitchMappingForExcludedSwitches();
    void buildPartitionsWithExcludedSwitch();
    void getPartitionSetForExcludedSwitch(SwitchKeyType key, PartitionSet &partitions);
    
    // Functions used by MODS during StagedInit
    void initializeAllLWLinks(void);
    void discoverAllLWLinks(void);
    void trainLWLinkConns(FMLWLinkTrainType trainTo);
    void finishInitialization(void);
    bool validateLWLinkConns(GlobalFMLWLinkConnRepo &failedConnections);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void reinitLWLinks(GlobalFMLWLinkConnRepo &failedConnections);
    void printAllFailedConnsFomAndGradingValues(GlobalFMLWLinkConnRepo &failedConnections);
#endif

    void setNodeConfigError(uint32_t nodeId);
    void clearNodeConfigError(uint32_t nodeId);
    bool isNodeConfigErrorOclwred(uint32_t nodeId);
    bool isInitDone() { return mFabricManagerInitDone; }
    bool isParallelTrainingEnabled();
    lwSwitchArchType getSwitchArchType();
    bool isSingleBaseboard(uint32_t nodeId);
	bool isSimMode() { return mSimMode; }
    bool isNodeDegraded(uint32_t nodeId);
 
    FMFabricParser               *mpParser; // parsed fabric topology
    FMFabricConfig               *mpConfig; // Fabric configuration
    std::map <uint32_t, FMFabricNode*>  mvFabricNodes;   // represents peer fabric nodes
    GlobalFMFabricPartitionMgr      *mGfmPartitionMgr; // manages shared lwswitch fabric partitions
    GlobalFmHaMgr                   *mpHaMgr; // manager GFM HA
    GlobalFmDegradedModeMgr         *mpDegradedModeMgr; // manage device failures and degradation
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    GlobalFmMulticastMgr            *mpMcastMgr;
#endif
    set<uint32_t> mDegradedNodes;

private:
    void doGlobalFMInitialization(GlobalFmArgs_t *pGfmArgs);
    void finishGlobalFMInitialization(void);
    void cleanup(void);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool isMultiNodeMode();
    FMIntReturn_t parseMultiNodeFabricTopology(const char *topoFile);
    void resetAndDrainLinks( GlobalFMLWLinkConnRepo &failedConnections );
#endif
    FMIntReturn_t parseNodeFabricTopology(const char *topoFile);
    void createFabricNodes(void);
    void createFabricNode(uint32_t nodeId, char *localFMIPAddress, char *domainSocketPath);
    bool waitForAllFabricNodeCtrlConns(void);
 
    void configureAllFabricNodes(void);
    void configureAllTrunkPorts(void);
    void sendGlobalConfigToAllFabricNodes(void);
    void sendPeerLFMInfoToAllFabricNodes(void);
    FMIntReturn_t sendInitDoneToAllFabricNodes(void);
    void sendDeInitToAllFabricNodes(void);
    static void processErrorWorkerRequest(job_t *pJob);
    void dumpAllLWSwitchInfo(void);
    void dumpAllGpuInfo(void);
    void dumpLWLinkDeviceAndInitStatus(void);
    void dumpGpuConnMatrix(void);
    void pruneNonDetectedLWSwitches(void);
    void pruneNonDetectedGpus(void);
    void disableLWSwitchTrunkLinks(void);
    
    void startNonSharedFabricMode(void);
    void startNonSharedInitialization(void);
    bool finishNonSharedInitialization();
    void degradeLWLinkDevices();

    void startSharedFabricMode(void);
    void startSharedInitialization(void);
    void finishSharedInitialization(void);
    
    void restartSharedFabricMode(void);
    
    FMIntReturn_t waitForAllFabricNodeConfigCompletion(void);
    void getExcludedConnectedSwitches( SwitchKeyType key, ConnectedSwitchesInfoMap &excludedConnectedSwitches);
    void createFmApiSocketInterafces(void);

    FMIntReturn_t sendLWSwitchTrainingFailedLinkInfoToAllNodes();
    FMIntReturn_t sendLWSwitchTrainingFailedLinkInfoToNode(uint32_t nodeId);
    FMIntReturn_t sendSwitchTrainingFailedLinksInfo(uint32_t nodeId, FMLWSwitchInfo &detectedSwitchInfo,
                                                    FMLWLinkDevInfo &switchLwLinkDevInfo);

    void checkFabricNodeVersions(void);

    bool isAllGpuLWLinksEnabled(uint32_t nodeId);
    void readFabricNodeConfigFile(char *fabricNodeConfigFile, std::map<uint32_t, string> &nodeToIpAddrMap);

    void doSingleNodeGlobalFMPreInit(GlobalFmArgs_t *pGfmArgs);
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void doMultiNodeGlobalFMPreInit(GlobalFmArgs_t *pGfmArgs);
#endif

    // context data while pushing to error worker queue
    typedef struct
    {
        GlobalFabricManager* pGfmObj;
        uint32 nodeId;
        lwswitch::fmMessage errorMsg;
    } FmErrorWorkerReqInfo;
 
    typedef std::list <FMMessageHandler*> MsgHandlerList;
    MsgHandlerList mMsgHandlers;
 
    // all the message handlers
    // link training interface message handler
    GlobalFMLWLinkIntf        *mLinkTrainIntf;
    // Various Device info message handler
    GlobalFMDevInfoMsgHdlr        *mDevInfoMsgHndlr;
    GlobalFmErrorStatusMsgHndlr *mGlobalErrStatusMsgHndl;
    GlobalFMLibCmdServer *mGlobalFMLibCmdServer;
    GlobalFMInternalCmdServer *mGlobalFMInternalCmdServer;
    GlobalFmApiHandler *mGlobalFmApiHandler;
 
    GlobalFMLWLinkDevRepo    mLWLinkDevRepo;
    GlobalFMLWLinkConnRepo    mLWLinkConnRepo;
    FMTopologyValidator *mTopoValidator;
    FMGpuInfoMap        mGpuInfoMap;
    FMGpuLWLinkSpeedInfoMap mGpuLWLinkSpeedInfoMap;
    FMExcludedGpuInfoMap        mExcludedGpuInfoMap;
    FMLWSwitchInfoMap   mLwswitchInfoMap;
    FMExcludedLWSwitchInfoMap mExcludedLwswitchInfoMap;
    GlobalFMCommandServer *mGlobalCmdServer;
    FMGpuLWLinkConnMatrixMap mGpuLWLinkConnMatrix;

    std::map<SwitchKeyType, ConnectedSwitchesInfoMap> mExcludedToConnectedSwitchInfoMap;
    std::map<SwitchKeyType, PartitionSet> mExcludedSwitchPartitionInfoMap;
 
    LWOSCriticalSection mLock;

    workqueue_t mErrWorkQueue;
    // starting TCP port number
    // mStartingPort for control connection between GFM and LFM
    // (mStartingPort + 1) for control connection between LFM and LFM    
    unsigned short mStartingPort;  
    unsigned int mFabricMode;
    bool mFabricManagerInitDone;
    bool mFabricModeRestart;
    bool mStagedInit;
    char *mFmLibCmdBindInterface;
    char *mFmLibCmdSockPath;
    unsigned short mFmLibPortNumber;
    bool mContinueWithFailure;
    bool mEnableTopologyValidation;
    lwSwitchArchType mSwitchArchType;
    bool mDisableDegradedMode;
    bool mDisablePruning;
    int mGfmWaitTimeout;
	bool mSimMode;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    char *mFabricNodeConfigFile;
    char *mMultiNodeTopology;
#endif
    int mFmLWlinkRetrainCount;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool mDisableLwlinkAli;
#endif
};
 
#endif
