
#pragma once
#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMLwcmClient.h"
#include "DcgmFMLwcmServer.h"

/*****************************************************************************/
/*  Local Fabric Managers cooperate for link training and memory management  */      
/*  There may or may not be a socket in each direction between any given     */
/*  pair of nodes, as either may be the master (client) or slave (server)    */
/*****************************************************************************/
class DcgmFMLocalCoOpMgr;
class DcgmLocalFabricManagerControl;
class DcgmFMLwcmClient;
class DcgmFMLwcmServer;

class DcgmFMLocalCoOpServer : public DcgmFMLwcmServer
{
    friend class DcgmFMLocalCoOpMgr;

private:
    DcgmFMLocalCoOpServer(DcgmFMConnBase* pConnBase, char *ip, unsigned short portNumber,
                          DcgmFMLocalCoOpMgr* parent);

    ~DcgmFMLocalCoOpServer();

     /*****************************************************************************
     This method handles the message received on the socket.
     *****************************************************************************/
    virtual int OnRequest(dcgm_request_id_t requestId, LwcmServerConnection *pConnection);

    DcgmFMLocalCoOpMgr* mParent;
};   

class DcgmFMLocalCoOpClientConn : public DcgmFMConnBase
{
    friend class DcgmFMLocalCoOpMgr;

private:
    DcgmFMLocalCoOpClientConn(char* host, unsigned short port, DcgmFMLocalCoOpMgr* parent);
    ~DcgmFMLocalCoOpClientConn();
    dcgmReturn_t SendMessage(lwswitch::fmMessage* pFmMessage, bool trackReq);

    // implementation of DcgmFMConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage* pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage * pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);

    DcgmFMLwcmClient   *mpFMClient;
    DcgmFMLocalCoOpMgr *mParent;
};

// This object act as a common place holder for all the LFM-LFM communication infra
class DcgmFMLocalCoOpMgr : public FMMessageHandler
{
public:
    DcgmFMLocalCoOpMgr(DcgmLocalFabricManagerControl* dcgmLFMHndle, 
                       char *coOpIp, unsigned short coOpPortNum);
    ~DcgmFMLocalCoOpMgr();

    void setLocalNodeId(uint32 nodeId);

    // FMMessageHandler overrides for handling FM messages from GFM
    virtual void handleMessage(lwswitch::fmMessage * pFmMessage);
    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    dcgmReturn_t SendMessageToPeerLFM(uint32 nodeId, lwswitch::fmMessage* pFmMessage, bool trackReq);

    int ProcessPeerNodeRequest(std::string nodeIpAddr, lwswitch::fmMessage* pFmMessage, bool &isResponse);

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
    typedef std::map <uint32, DcgmFMLocalCoOpClientConn*> NodeIdClientConnMap;
    NodeIdClientConnMap mNodeIdClientConnMap;

    DcgmLocalFabricManagerControl* mDcgmLFMHndle; // TODO make this as DcgmLocalFabricManager pointer after re-org LFM code
    unsigned short mCoOpPortNum;
    uint32 mSelfNodeId;

    // listening server object for incoming connection
    DcgmFMLocalCoOpServer* mCoOpServer;
    bool mCoOpServerStarted;
};

