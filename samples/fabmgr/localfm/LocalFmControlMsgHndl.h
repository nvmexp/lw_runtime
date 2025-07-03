#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "FMCommonTypes.h"
#include "fabricmanager.pb.h"
#include "LocalFabricManager.h"
#include "FMCommCtrl.h"
#include "workqueue.h"
#include <g_lwconfig.h>

class LocalFMControlMsgHndl : public FMMessageHandler
{
public:
    LocalFMControlMsgHndl(LocalFabricManagerControl *pControl);

    virtual ~LocalFMControlMsgHndl();

    virtual void handleEvent( FabricManagerCommEventType eventType, uint32 nodeId );

    void virtual handleMessage( lwswitch::fmMessage *pFmMessage );
    void virtual dumpMessage( lwswitch::fmMessage *pFmMessage );
    void colwertDriverToFmDegradedReason(lwswitch::SwitchDegradedReason &reason, 
                                         LWSWITCH_DEVICE_BLACKLIST_REASON excludedReason);
    static void gfmHeartbeatTimeoutHandler(void *);
    void gfmHeartbeatTimeoutProcess();

private:
    static void processMessageJob( job_t *pJob );
    void processMessage( lwswitch::fmMessage *pFmMessage );

    void handleNodeGlobalConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleSWPortConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleIngReqTblConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleIngRespTblConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGangedLinkTblConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuConfigReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuAttachReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuDetachReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigInitDoneReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleConfigDeInitReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleHeartbeatMsg( lwswitch::fmMessage *pFmMessage );

    void handleSwitchDisableLinkReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGpuSetDisabledLinkMaskReqMsg( lwswitch::fmMessage *pFmMessage );

    void handleGetGfidReqMsg(lwswitch::fmMessage *pFmMessage);

    void handleCfgGfidReqMsg(lwswitch::fmMessage *pFmMessage);

    void handlePortRmapConfigReqMsg( lwswitch::fmMessage *pFmMessage );
    void handlePortRidConfigReqMsg( lwswitch::fmMessage *pFmMessage );
    void handlePortRlanConfigReqMsg( lwswitch::fmMessage *pFmMessage );
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    bool mcIdEntryToIoctlParams( uint32_t portNum, bool noDynRsp,
                                 const lwswitch::MulticastTableEntry &entry,
                                 LWSWITCH_SET_MC_RID_TABLE_PARAMS &ioctlParams );
    void handleMulticastIdRequestMsg( lwswitch::fmMessage *pFmMessage );
#endif

    void handleDegradedGpuInfoMsg( lwswitch::fmMessage *pFmMessage );
    void handleDegradedSwitchInfoMsg( lwswitch::fmMessage *pFmMessage );

    void colwertFmToDriverDegradedReason(lwswitch::SwitchDegradedReason reason,
                                         LWSWITCH_DEVICE_BLACKLIST_REASON &excludedReason);

    void colwertFmToDriverRemapPolicyFlags(uint32_t fmFlags, uint32_t &driverFlags);

    FMIntReturn_t SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );

    LocalFabricManagerControl  *mpControl;            // server connection from lfm to gfm
    std::map <uint32_t, workqueue_t *> mvWorkqueue;   // work queues for processing message
    FMTimer *mGfmHeartbeatTimer;                      // If heartbeats from GFM stop for 
                                                      // (HEARTBEAT_INTERVAL * HEARTBEAT_THRESHOLD) seconds close
                                                      // FMSession and stop processing IMEX message
};

/**
 * Structure for FM message user data to push to worker queue
 */
typedef struct FmMessageUserData
{
    lwswitch::fmMessage     *pFmMessage; /* Represents FM message  */
    LocalFMControlMsgHndl   *pHndl;      /* Represents LocalFMControlMsgHndl object */
} FmMessageUserData_t;

