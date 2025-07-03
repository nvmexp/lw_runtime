
#pragma once
#include "fabricmanager.pb.h"
#include "FMCommonTypes.h"
#include "FMCommCtrl.h"
#include "FMLwcmClient.h"
#include "LocalFMLwcmServer.h"
#include <set>

/*****************************************************************************/
/*  Local Fabric Managers cooperate for link training and memory management  */      
/*  There may or may not be a socket in each direction between any given     */
/*  pair of nodes, as either may be the master (client) or slave (server)    */
/*****************************************************************************/
class LocalFMCoOpMgr;
class LocalFabricManagerControl;
class FMLwcmClient;
class LocalFMLwcmServer;

class LocalFMCoOpServer : public LocalFMLwcmServer
{
    friend class LocalFMCoOpMgr;

private:
    LocalFMCoOpServer(FmConnBase* pConnBase, char *ip, unsigned short portNumber,
                      LocalFMCoOpMgr* parent, uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold);

    ~LocalFMCoOpServer();

     /*****************************************************************************
     This method handles the message received on the socket.
     *****************************************************************************/
    virtual int OnRequest(fm_request_id_t requestId, FmServerConnection *pConnection);

    LocalFMCoOpMgr* mParent;
};   

class LocalFMCoOpClientConn : public FmConnBase
{
    friend class LocalFMCoOpMgr;

private:
    LocalFMCoOpClientConn(char* host, unsigned short port, LocalFMCoOpMgr* parent,
                          uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold);
    ~LocalFMCoOpClientConn();
    FMIntReturn_t SendMessage(lwswitch::fmMessage* pFmMessage, bool trackReq);

    // implementation of FmConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage* pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);

    FMLwcmClient   *mpFMClient;
    LocalFMCoOpMgr *mParent;
};

// This object act as a common place holder for all the LFM-LFM communication infra
class LocalFMCoOpMgr : public FMMessageHandler
{
public:
    LocalFMCoOpMgr(LocalFabricManagerControl* pLFMHndle, 
                   char *coOpIp, unsigned short coOpPortNum);
    ~LocalFMCoOpMgr();

    void setLocalNodeId(uint32 nodeId);

    // FMMessageHandler overrides for handling FM messages from GFM
    virtual void handleMessage(lwswitch::fmMessage * pFmMessage);
    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    FMIntReturn_t SendMessageToPeerLFM(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool trackReq);

    int ProcessPeerNodeRequest(std::string nodeIpAddr, lwswitch::fmMessage* pFmMessage, bool &isResponse);

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void getPeerNodeIds(std::set<uint32> &nodeIds);
#endif

private:

    void startCoOpServer(void);
    void ProcessNodeInfoMsg(lwswitch::fmMessage* pFmMessage);
    void CleanupPeerConnections(void);
    void CreatePeerConnections(void);
    void SendNodeInfoMsgAck(lwswitch::fmMessage* pFmReqMessage);
    
    // maintains all the node's nodeId to IP address mapping
    typedef std::map <uint32, std::string> NodeIdIpAddrMap;
    NodeIdIpAddrMap mNodeIdIpAddrMap;

    // maintains nodeId to Client Connection mapping, ie connection initiated by this node
    typedef std::map <uint32, LocalFMCoOpClientConn*> NodeIdClientConnMap;
    NodeIdClientConnMap mNodeIdClientConnMap;

    LocalFabricManagerControl* mpLFMHndle;
    unsigned short mCoOpPortNum;
    uint32 mSelfNodeId;

    // listening server object for incoming connection
    LocalFMCoOpServer* mCoOpServer;
    bool mCoOpServerStarted;
};

