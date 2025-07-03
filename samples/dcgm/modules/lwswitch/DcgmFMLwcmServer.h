
#pragma once

#include "dcgm_structs.h"
#include "fabricmanager.pb.h"
#include "LwcmServerConnection.h"

#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmFMConnectionBase.h"

class DcgmFMLwcmServerConn : public DcgmFMConnectionBase
{

public:
    DcgmFMLwcmServerConn(DcgmFMConnBase* parent, LwcmServerConnection *pConnection);

    virtual ~DcgmFMLwcmServerConn();

    struct sockaddr_in getRemoteSocketAddr();
};

class DcgmFMLwcmServer : public LwcmServer
{

public:
    DcgmFMLwcmServer( DcgmFMConnBase* parent, int portNumber, char *sockpath, int isTCP );
    virtual ~DcgmFMLwcmServer();

    /*****************************************************************************
    This method handles the message received on the socket.
    *****************************************************************************/
   virtual int OnRequest(dcgm_request_id_t requestId, LwcmServerConnection *pConnection);

   virtual void OnConnectionAdd(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);
    
   virtual void OnConnectionRemove(dcgm_connection_id_t connectionId, LwcmServerConnection *pConnection);

   // send message to a specific IP address
   virtual dcgmReturn_t sendFMMessage(std::string ipAddr, lwswitch::fmMessage* pFmMessage, bool trackReq);

   // send message to the first connection or the only connection
   virtual dcgmReturn_t sendFMMessage(lwswitch::fmMessage* pFmMessage, bool trackReq);

   // close all the server connection accepted by this node
   virtual void closeAllServerConnections(void);

   DcgmFMLwcmServerConn *getFirstConn();
   DcgmFMLwcmServerConn *getFMServerConn(std::string ipAddr);

protected:
   // maintains a list of DcgmFMLwcmServerConn mapping
   typedef std::map<dcgm_connection_id_t, DcgmFMLwcmServerConn *> ConnIdToFMServerConnMap;
   ConnIdToFMServerConnMap mFMServerConnMap;

   DcgmFMConnBase* mpParent;
   int mPortNumber;
};
