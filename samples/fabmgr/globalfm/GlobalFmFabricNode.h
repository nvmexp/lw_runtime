/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
 #pragma once

#include <signal.h>
#include <time.h>
#include <pthread.h>

#include "fabricmanager.pb.h"
#include "FMCommonTypes.h"
#include "FMCommCtrl.h"
#include "lwos.h"


/*****************************************************************************/
/*  Abstract a peer Fabric Node in Global Fabric Manager                     */
/*****************************************************************************/

class GlobalFabricManager;
class FMGlobalHeartbeat;
class FMLwcmClient;

class FMFabricNode : public FmConnBase
{
public:
    FMFabricNode(const char *identifier, uint32 nodeId,
                   GlobalFabricManager *pGfm,
                   bool addressIsUnixSocket);
    ~FMFabricNode();

    FMIntReturn_t SendControlMessage(lwswitch::fmMessage *message, bool trackReq);
    FMIntReturn_t SendControlMessageSync(lwswitch::fmMessage *pFmMessage,
                                         lwswitch::fmMessage **pResponse,
                                         uint32_t timeoutSec);

    // implementation of FmConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage *pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage *pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);
    bool isControlConnectionActive(void);

    uint32 getNodeId ( void ) { return mNodeId; }
    std::string getNodeAddress() { return mNodeAddress; }
    uint32_t getControlMessageRequestId( void );

    void setConfigError(void);
    void clearConfigError(void);
    bool isConfigError(void);
    void processHeartBeatFailure();

private:
    uint32 mNodeId;
    LWOSCriticalSection mLock;
    GlobalFabricManager *mpGfm;
    FMLwcmClient *mpClientConn;
    FMGlobalHeartbeat *mpHeartbeat;
    bool mConfigError;
    std::string mNodeAddress;
};
