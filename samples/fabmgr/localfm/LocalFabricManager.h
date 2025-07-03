#pragma once
 
#include <signal.h>
#include "fm_config_options.h"
#include "fabricmanager.pb.h"
#include "fmlib.pb.h"
#include "LocalFabricManagerCoOp.h"
#include "LocalFMSwitchInterface.h"
#include "LocalFmControlMsgHndl.h"
#include "LocalFMLWLinkMsgHndlr.h"
#include "LocalFMLWLinkDrvIntf.h"
#include "FMLWLinkDeviceRepo.h"
#include "FMDevInfoMsgHndlr.h"
#include "LocalFMLwcmServer.h"
#include "LocalFMGpuMgr.h"
#include "FMCommonTypes.h"
#include "FMErrorCodesInternal.h"
#include <g_lwconfig.h>

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
#include "LocalFmMulticastHndlr.h"
#endif

/*****************************************************************************/
/*  Local Fabric Manager is Server to Global Fabric Managers for topology    */      
/*  discovery and configuration, heartbeat, and error notification           */
/*  Also server or client to other Local Fabric Managers for link traininng, */
/*  and memory management coordination across nodes.                         */
/*****************************************************************************/
 
// Local Fabric Manager is a client of the switch interface, and uses request IDs to tag
// and query its IOCTL requests. Because there are more than one client ilwolved, each 
// with their own connections (and because the request IDs are just an atomic count per  
// connection, we need to keep the request IDs unique by forcing high order bits of the ID 
// to different values per client. Lwrrently allowing for up to 16 clients.
 
#define LOCAL_REQUESTID_CLIENT_MASK         0x0FFFFFFF
#define LOCAL_REQUESTID_CLIENT_CONTROL      0x00000000
#define LOCAL_REQUESTID_CLIENT_HEARTBEAT    0x10000000
#define LOCAL_REQUESTID_CLIENT_COOP         0x20000000
 
class LocalFMErrorReporter;
class LocalFMControlMsgHndl;
class LocalFMLwcmServer;
class LocalFMCommandServer;
class LocalFMGpuMgr;
class LocalFMSwitchEventReader;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
class LocalFMMemMgr;
class LocalFMMemMgrExporter;
class LocalFMMemMgrImporter;
#endif
class LocalFMErrorReporter;
class LocalFmSwitchHeartbeatReporter;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
class LocalFmMulticastHndlr;
#endif

/*****************************************************************************/
/*  Control is the server that responds to requests from the Global Fabric   */      
/*  manager.                                                                 */
/*****************************************************************************/
 
typedef struct {
    unsigned int fabricMode;
    char *bindInterfaceIp;
    unsigned short fmStartingTcpPort;
    char *domainSocketPath;
    bool continueWithFailures;
    unsigned int abortLwdaJobsOnFmExit;
    unsigned int switchHeartbeatTimeout;
    bool simMode;
    unsigned int imexReqTimeout;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool disableLwlinkAli;
#endif
} LocalFmArgs_t;
 
class LocalFabricManagerControl : public FmConnBase, FMConnInterface
{
    friend class LocalFMCommandServer;
    friend class LocalFMGpuMgr;
    friend class LocalFMErrorReporter;

public:
    //
    // we only need to track whether we allocated or deallocated FM session
    // No need to track set/clear state. Also FM session state set/clear is stateless in RM also.
    //
    typedef enum FmSessionState {
        STATE_NOT_ALLOCATED = 0,
        STATE_ALLOCATED,
    } FmSessionState_t;
 
    LocalFabricManagerControl(LocalFmArgs_t *lfm);
 
    ~LocalFabricManagerControl();
    LocalFMSwitchInterface *switchInterfaceAt( uint32_t physicalId );
    LocalFMSwitchInterface *switchInterfaceAt( FMUuid_t &uuid );
    LocalFMSwitchInterface *switchInterfaceAtIndex( int index );
    uint32_t getSwitchDevIndex( uint32_t physicalId );
 
    int ProcessPeerLFMMessage(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool &isResponse);
 
    // implementation of FmConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);
 
    // implementation of FMConnInterface pure virtual functions
    virtual FMIntReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );
    virtual FMIntReturn_t SendMessageToLfm( uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage, bool trackReq );
 
    void setLocalNodeId(uint32 nodeId);
    uint32 getLocalNodeId( ) { return mMyNodeId; }
    uint32 getNumLwswitchInterface( ) { return mvSwitchInterface.size(); }
    void getAllLWLinkDevInfo(FMLWLinkDevInfoList &lwlinkDevList);
    void getAllLwswitchInfo(FMLWSwitchInfoList &switchInfoList);
    void getExcludedLwswitchInfo(FMExcludedLWSwitchInfoList &excludedLwswitchInfoList);
    void addDegradedSwitchInfo(uint32_t physicalId, lwswitch::SwitchDegradedReason reason);
    bool isSwitchDegraded(uint32_t physicalId);
    void getGpuPciInfo(FMUuid_t &uuid, FMPciInfo_t &pciInfo)    ;
    void getAllGpuInfo(FMGpuInfoList &gpuInfoList);
    void getExcludedGpuInfo(FMExcludedGpuInfoList &excludedGpuInfoList);
    bool onConfigInitDoneReqRcvd(void);
    bool onConfigDeInitReqRcvd(void);
    unsigned int getFabricMode( ) { return mFabricMode; }
    bool isSimMode() { return mSimMode; }

    FMIntReturn_t detachAllGpus( void );
    FMIntReturn_t attachGpu( FMUuid_t uuid, bool registerEvent );
    FMIntReturn_t detachGpu( FMUuid_t uuid, bool unregisterEvent );
    FMIntReturn_t refreshRmLibProbedGpuInfo( void );
    FMIntReturn_t getGpuGfid(FMUuid_t uuid, FMPciInfo_t &vf, uint32_t &gfid, uint32_t &gfidMask);
    FMIntReturn_t configGpuGfid(FMUuid_t uuid, uint32_t gfid, bool activate);
    void closeAndDeleteLWSwitchInterface(uint32_t physicalId);

    void handleStayResidentCleanup(void);
    void setFmDriverStateToStandby(void);

    LocalFMGpuMgr *mFMGpuMgr;

private:
    void doLocalFMInitialization(LocalFmArgs_t *lfm);
    void cleanup(void);

    int ProcessRequest(lwswitch::fmMessage *pFmMessage, FmServerConnection *pConnection, bool *pIsComplete);
 
    int HandleCommands(vector<fmlib::Command *> *pVecCmdsToProcess, 
                       bool *pIsComplete);
 
    void createLwswitchInterface(void);

    void addExcludedLwSwitchInfo(LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo);
 
    bool freeFmSession(void);

    LocalFMSwitchEventReader *mLocalFMSwitchEvtReader;
 
    LocalFMLwcmServer *mpControlServer;

    std::vector <LocalFMSwitchInterface *> mvSwitchInterface;
    std::map <int, lwswitch::SwitchDegradedReason> mDegradedSwitchInfo;
    FMExcludedLWSwitchInfoList mExcludedLwswitchInfoList;

    LocalFMControlMsgHndl *mpControlMsgHndl;
 
    LocalFMCoOpMgr* mLocalFMCoOpMgr;
    
 
    LocalFMLWLinkDrvIntf *mLWLinkDrvIntf;
    LocalFMLWLinkDevRepo* mLWLinkDevRepo;
    LocalFMLWLinkMsgHndlr* mLinkTrainHndlr;
    LocalFMDevInfoMsgHdlr* mDevInfoMsgHndlr;
    LocalFMCommandServer *mLocalCmdServer;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    LocalFMMemMgr *mLocalFMMemMgr;
    LocalFMMemMgrExporter *mLocalFMMemMgrExporter;
	LocalFMMemMgrImporter *mLocalFMMemMgrImporter;
#endif
    LocalFmSwitchHeartbeatReporter *mLocalFmSwitchHeartbeatReporter;
 
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LocalFmMulticastHndlr* mMulticastHndlr;
#endif

    unsigned int mMyNodeId;
    FmSessionState_t mFmSessionState;
    unsigned int mFabricMode;
    bool mContinueWithFailure;
    unsigned int mAbortLwdaJobsOnFmExit;
    unsigned int mSwitchHeartbeatTimeout;
    unsigned int mImexReqTimeout;
    bool mSimMode;
};   

