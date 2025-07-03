
#ifndef DCGMGLOBALFABRICMANAGER_H
#define DCGMGLOBALFABRICMANAGER_H

#include <queue>
#include <unistd.h>
#include "workqueue.h"
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMError.h"

#include "DcgmFabricConfig.h"
#include "DcgmFMLwcmClient.h"
#include "DcgmFMLWLinkIntf.h"
#include "DcgmFMLWLinkTypes.h"
#include "DcgmFMLWLinkConnRepo.h"
#include "DcgmFMDevInfoMsgHndlr.h"
#include "DcgmFabricNode.h"
#include "LwcmCacheManager.h"
#include "LwcmHostEngineHandler.h"
#include "dcgm_module_fm_structs_internal.h"

/*****************************************************************************/
/*  Global Fabric Manager is client to Local Fabric Managers for topology    */      
/*  discovery and configuration, heartbeat, and query relay to other Local   */
/*  Fabric Managers.                                                         */
/*  Also server to DCGM for topology and perf queries, resource isolation,   */
/*  and error notification.                                                  */
/*****************************************************************************/

class DcgmFabricConfig;
class DcgmFabricParser;
class DcgmGlobalControlMsgHndl;
class DcgmGlobalFabricManager;
class DcgmGlobalStatsMsgHndlr;
class DcgmFMTopologyValidator;
class DcgmGlobalCommandServer;
class DcgmGlobalFMErrorHndlr;
class DcgmGFMFabricPartitionMgr;
class DcgmGlobalFabricManagerHaMgr;

#define HEARTBEAT_INTERVAL    10 // second

typedef struct {
    uint32_t gpuPhyIndex;
    uint32_t gpuEnumIndex;
    uint64 enabledLinkMask;
    bool linkConnStatus[DCGM_LWLINK_MAX_LINKS_PER_GPU];
} DcgmFMGpuLWLinkConnMatrix;

typedef std::list<DcgmFMGpuLWLinkConnMatrix> DcgmFMGpuLWLinkConnMatrixList;
typedef std::map<uint32, DcgmFMGpuLWLinkConnMatrixList>  DcgmFMGpuLWLinkConnMatrixMap;

class DcgmGlobalFabricManager: public FMConnInterface
{
    friend class DcgmGlobalCommandServer;
    friend class DcgmFMTopologyValidator;
    friend class DcgmGlobalFMErrorHndlr;
    friend class DcgmFMLWLinkDevInfo;
    friend class DcgmGFMFabricPartitionMgr;

public:
    DcgmGlobalFabricManager(bool sharedFabric,
                            bool restart,
                            unsigned short startingPort,
                            char *domainSocketPath,
                            char *stateFilename);
    ~DcgmGlobalFabricManager();

    /*****************************************************************************/
    // these methods will be called as part of DCGM module command handling
    dcgmReturn_t getSupportedFabricPartitions(dcgmFabricPartitionList_t &dcgmFabricPartitions);
    dcgmReturn_t activateFabricPartition(unsigned int partitionId);
    dcgmReturn_t deactivateFabricPartition(unsigned int partitionId);
    dcgmReturn_t setActivatedFabricPartitions(dcgmActivatedFabricPartitionList_t &dcgmFabricPartitions);
    /*****************************************************************************/

    int ProcessMessage(uint32 nodeId, lwswitch::fmMessage * pFmMessage, bool &isResponse);
    void OnFabricNodeConnect(uint32 nodeId);
    void OnFabricNodeDisconnect(uint32 nodeId);

    // implementation of FMConnInterface pure virtual functions
    virtual dcgmReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );
    virtual dcgmReturn_t SendMessageToLfm( uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage,
                                           bool trackReq );
    virtual uint32_t getControlMessageRequestId(uint32_t fabricNodeId);

    void getNodeErrors(uint32_t nodeId, uint32_t errorMask);
    void getNodeStats(uint32_t nodeId );
    void queueErrorWorkerRequest(uint32 nodeId, lwswitch::fmMessage *errorMsg);
    bool getGpuPhysicalIndex(uint32 nodeId, uint32_t enumIndex, uint32_t &physicalIdx);
    bool getGpuEnumIndex(uint32 nodeId, uint32_t physicalIdx, uint32_t &enumIndex);
    bool getGpuPciBdf(uint32 nodeId, uint32_t enumIndex, DcgmFMPciInfo &pciInfo);
    bool getGpuUuid(uint32 nodeId, uint32 physicalId, char uuid[]);
    bool getLWSwitchLWLinkDriverId(uint32 nodeId, uint32 physicalId, uint64 &lwLinkSwitchId);

    unsigned short getStartingPort( ) { return mStartingPort; }
    bool isSharedFabricMode( ) { return mSharedFabric; }
    bool isRestart() { return mRestart; }

    const DcgmFMGpuInfoMap& getGpuInfoMap() const { return mGpuInfoMap; }
    const DcgmFMGpuInfoMap& getBlacklistGpuInfoMap() const { return mBlacklistGpuInfoMap; }    
    const DcgmFMLWSwitchInfoMap& getLwSwitchInfoMap() const { return mLwswitchInfoMap; }
    DcgmCacheManager* GetCacheManager() const { return mpCacheManager; }

    void setNodeConfigError(uint32_t nodeId);
    void clearNodeConfigError(uint32_t nodeId);
    bool isNodeConfigErrorOclwred(uint32_t nodeId);

    DcgmFabricParser               *mpParser; // parsed fabric topology
    std::map <uint32_t, DcgmFabricNode*>  mvFabricNodes;   // represents peer fabric nodes
    DcgmGFMFabricPartitionMgr      *mGfmPartitionMgr; // manages shared lwswitch fabric partitions    
    DcgmGlobalFabricManagerHaMgr   *mpHaMgr; // manager GFM HA

private:
    FM_ERROR_CODE parseFabric();
    void createFabricNodes(char *domainSocketPath);
    bool waitForAllFabricNodeCtrlConns(void);

    void configureAllFabricNodes(void);
    void configureAllTrunkPorts(void);
    void sendGlobalConfigToAllFabricNodes(void);    
    void sendInitDoneToAllFabricNodes(void);
    void sendDeInitToAllFabricNodes(void);
    static void processErrorWorkerRequest(job_t *pJob);
    void dumpAllLWSwitchInfo(void);
    void dumpAllGpuInfo(void);
    void dumpLWLinkDeviceAndInitStatus(void);
    void dumpGpuConnMatrix(void);
    void addSwitchesToCacheManager(void);
    void publishLinkStatetoCacheManager(void);
    void pruneNonDetectedLWSwitches(void);
    void pruneNonDetectedGpus(void);
    void disableLWSwitchTrunkLinks(void);
    void initializeAllLWLinksAndTrainConns(void);
    void startNonSharedFabricMode(void);
    void startSharedFabricMode(void);
    void restartSharedFabricMode(void);
    FM_ERROR_CODE waitForAllFabricNodeConfigCompletion(void);

    // context data while pushing to error worker queue
    typedef struct
    {
        DcgmGlobalFabricManager* pGfmObj;
        uint32 nodeId;
        lwswitch::fmMessage errorMsg;
    } FmErrorWorkerReqInfo;

    typedef std::list <FMMessageHandler*> MsgHandlerList;
    MsgHandlerList mMsgHandlers;

    // all the message handlers
    // Fabric configuration
    DcgmFabricConfig        *mpConfig;
    // link training interface message handler
    DcgmFMLWLinkIntf        *mLinkTrainIntf;
    // Various Device info message handler
    GFMDevInfoMsgHdlr        *mDevInfoMsgHndlr;
    DcgmGlobalControlMsgHndl *mControlMsgHndl;
    DcgmGlobalStatsMsgHndlr *mGlobalStatsMsgHndl;

    DcgmGFMLWLinkDevRepo    mLWLinkDevRepo;
    DcgmFMLWLinkConnRepo    mLWLinkConnRepo;
    DcgmFMTopologyValidator *mTopoValidator;
    DcgmFMGpuInfoMap        mGpuInfoMap;
    DcgmFMGpuInfoMap        mBlacklistGpuInfoMap;
    DcgmFMLWSwitchInfoMap   mLwswitchInfoMap;
    DcgmGlobalCommandServer *mGlobalCmdServer;
    DcgmFMGpuLWLinkConnMatrixMap mGpuLWLinkConnMatrix;

    LwcmHostEngineHandler *mpHostEngineHandler;
    DcgmCacheManager *mpCacheManager;

    LWOSCriticalSection mLock;
    workqueue_t mErrWorkQueue;
    // starting TCP port number
    // mStartingPort for control connection between GFM and LFM
    // (mStartingPort + 1) for control connection between LFM and LFM    
    unsigned short mStartingPort;  
    bool mSharedFabric;
    bool mFabricManagerInitDone;
    bool mRestart;
};


#endif
