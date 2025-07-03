#pragma once

#include <signal.h>
#include <time.h>
#include <pthread.h>

#include "fabricmanager.pb.h"
#include "DcgmFMCommon.h"
#include "DcgmFMCommCtrl.h"
#include "lwos.h"


/*****************************************************************************/
/*  Abstract a peer Fabric Node in Global Fabric Manager                     */
/*****************************************************************************/

class DcgmGlobalFabricManager;
class DcgmGlobalHeartbeat;
class DcgmFMLwcmClient;

class DcgmFabricNode : public DcgmFMConnBase
{
public:
    DcgmFabricNode(const char *identifier, uint32 nodeId,
                   DcgmGlobalFabricManager *dcgmGFMHndle,
                   bool addressIsUnixSocket);
    ~DcgmFabricNode();

    dcgmReturn_t SendControlMessage(lwswitch::fmMessage *message, bool trackReq);
    dcgmReturn_t SendControlMessageSync(lwswitch::fmMessage *pFmMessage, lwswitch::fmMessage **pResponse);

    // implementation of DcgmFMConnBase pure virtual functions
    virtual int ProcessMessage(lwswitch::fmMessage *pFmMessage, bool &isResponse);
    virtual void ProcessUnSolicitedMessage(lwswitch::fmMessage *pFmMessage);
    virtual void ProcessConnect(void);
    virtual void ProcessDisconnect(void);
    bool isControlConnectionActive(void);

    uint32 getNodeId ( void ) { return mNodeId; }
    uint32_t getControlMessageRequestId( void );

    void setConfigError(void);
    void clearConfigError(void);
    bool isConfigError(void);

private:
    uint32 mNodeId;
    LWOSCriticalSection mLock;
    DcgmGlobalFabricManager *mDcgmGFMHndle;
    DcgmFMLwcmClient *mpClientConn;
    DcgmGlobalHeartbeat *mpHeartbeat;
    bool mConfigError;
};
