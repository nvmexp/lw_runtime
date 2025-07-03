/*
 *  Copyright 2018-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once

#include "fabricmanager.pb.h"
#include "FmServerConnection.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "FMCommCtrl.h"
#include "FMConnectionBase.h"

class FMServerConn : public FMConnectionBase
{

public:
    FMServerConn(FmConnBase* parent, FmServerConnection *pConnection,
                 uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold);

    virtual ~FMServerConn();

    struct sockaddr_in getRemoteSocketAddr();
};

class LocalFMLwcmServer : public FmSocket
{

public:
    LocalFMLwcmServer( FmConnBase* parent, int portNumber, char *sockpath, int isTCP,
                       uint32_t rspTimeIntrvl, uint32_t rspTimeThreshold );
    virtual ~LocalFMLwcmServer();

    /*****************************************************************************
    This method handles the message received on the socket.
    *****************************************************************************/
   virtual int OnRequest(fm_request_id_t requestId, FmServerConnection *pConnection);

   virtual void OnConnectionAdd(fm_connection_id_t connectionId, FmServerConnection *pConnection);
    
   virtual void OnConnectionRemove(fm_connection_id_t connectionId, FmServerConnection *pConnection);

   // send message to a specific IP address
   virtual FMIntReturn_t sendFMMessage(std::string ipAddr, lwswitch::fmMessage* pFmMessage, bool trackReq);

   // send message to the first connection or the only connection
   virtual FMIntReturn_t sendFMMessage(lwswitch::fmMessage* pFmMessage, bool trackReq);

   // close all the server connection accepted by this node
   virtual void closeAllServerConnections(void);

   FMServerConn *getFirstConn();
   FMServerConn *getFMServerConn(std::string ipAddr);

protected:
   // maintains a list of FMServerConn mapping
   typedef std::map<fm_connection_id_t, FMServerConn *> ConnIdToFMServerConnMap;
   ConnIdToFMServerConnMap mFMServerConnMap;

   LWOSCriticalSection mMutex;
   FmConnBase* mpParent;
   int mPortNumber;

   uint32_t mRspTimeIntrvl;
   uint32_t mRspTimeThreshold;
};
