#pragma once

#include <signal.h>
extern "C"
{
    #include "lwswitch_user_linux.h"
}
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmLocalFabricManagerCoOp.h"
#include "DcgmSwitchInterface.h"
#include "DcgmLocalControlMsgHndl.h"
#include "DcgmFMLWLinkMsgHndlr.h"
#include "DcgmFMLWLinkDrvIntf.h"
#include "DcgmFMLWLinkDeviceRepo.h"
#include "DcgmFMDevInfoMsgHndlr.h"
#include "DcgmFMLwcmServer.h"
#include "lwml_internal.h"
#include "LwcmHostEngineHandler.h"


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

class DcgmLocalStatsReporter;
class DcgmLocalStatsMsgHndlr;
class DcgmLocalControlMsgHndl;
class DcgmFMLwcmServer;
class DcgmLocalCommandServer;
class DcgmLocalMemMgr;

/*****************************************************************************/
/*  Control is the server that responds to requests from the Global Fabric   */      
/*  manager.                                                                 */
/*****************************************************************************/

class DcgmLocalFabricManagerControl : public DcgmFMConnBase, FMConnInterface
{
    friend class DcgmLocalCommandServer;

public:
    typedef enum FmSessionState {
        STATE_CLOSED = 0,
        STATE_ALLOCATED,
        STATE_SET
    } FmSessionState_t;

    DcgmLocalFabricManagerControl(bool sharedFabric, char *bindInterfaceIp, 
                                  unsigned short portNumber,
                                  char *domainSocketPath);

    ~DcgmLocalFabricManagerControl();

    bool QueryInitComplete();

    DcgmSwitchInterface *switchInterfaceAt( uint32_t physicalId );
    uint32_t getSwitchDevIndex( uint32_t physicalId );

    int ProcessPeerLFMMessage(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool &isResponse);

    // implementation of DcgmFMConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage * pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);

    // implementation of FMConnInterface pure virtual functions
    virtual dcgmReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );
    virtual dcgmReturn_t SendMessageToLfm( uint32 fabricNodeId, lwswitch::fmMessage *pFmMessage, bool trackReq );

    void setLocalNodeId(uint32 nodeId);
    uint32 getLocalNodeId( ) { return mMyNodeId; }
    uint32 getNumLwswitchInterface( ) { return mvSwitchInterface.size(); }
    void getAllLWLinkDevInfo(DcgmFMLWLinkDevInfoList &lwlinkDevList);
    void getAllLwswitchInfo(DcgmFMLWSwitchInfoList &switchInfoList);
    void getAllGpuInfo(DcgmFMGpuInfoList &gpuInfoList);
    void getBlacklistGpuInfo(DcgmFMGpuInfoList &blacklistGpuInfoList);
    bool onConfigInitDoneReqRcvd(void);
    bool onConfigDeInitReqRcvd(void);
    bool isSharedFabricMode( ) { return mSharedFabric; }

    dcgmReturn_t attachAllGpus( void );
    dcgmReturn_t detachAllGpus( void );

    DcgmCacheManager *mpCacheManager;

private:
    int ProcessRequest(lwswitch::fmMessage *pFmMessage, LwcmServerConnection *pConnection, bool *pIsComplete);

    int HandleCommands(vector<lwcm::Command *> *pVecCmdsToProcess, 
                       bool *pIsComplete);

    void createLwswitchInterface(void);

    bool freeFmSession(void);

    LwcmHostEngineHandler *mpHostEngineHandler;

    DcgmFMLwcmServer *mpControlServer;

    std::vector <DcgmSwitchInterface *> mvSwitchInterface;
    DcgmLWSwitchPhyIdToFdInfoMap mSwitchIdToFdInfoMap;

    DcgmLocalControlMsgHndl *mpControlMsgHndl;

    DcgmFMLocalCoOpMgr* mFMLocalCoOpMgr;
    DcgmLocalStatsReporter *mLocalStatsReporter;

    DcgmFMLWLinkDrvIntf *mLWLinkDrvIntf;
    DcgmLFMLWLinkDevRepo* mLWLinkDevRepo;
    DcgmFMLWLinkMsgHndlr* mLinkTrainHndlr;
    LFMDevInfoMsgHdlr* mDevInfoMsgHndlr;
    DcgmLocalStatsMsgHndlr* mLocalStatsMsgHndlr;
    etblLWMLCommonInternal_st * metblLwmlCommonInternal;
    DcgmLocalCommandServer *mLocalCmdServer;
    DcgmLocalMemMgr *mLocalMemMgr;

    unsigned int mMyNodeId;
    bool mInitComplete;
    FmSessionState_t mFmSessionState;
    bool mSharedFabric;
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    bool mIsMultiNode;
#endif
};   

class DcgmLocalFabricManager
{
public:
    DcgmLocalFabricManager(bool sharedFabric, char *bindInterfaceIp,
                           unsigned short startingPort,
                           char *domainSocketPath);
    ~DcgmLocalFabricManager();

private:
    DcgmLocalFabricManagerControl *mpGlobalControl;
};

