#ifndef DCGM_LOCAL_CONTROL_MSG_HNDL_H
#define DCGM_LOCAL_CONTROL_MSG_HNDL_H

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "DcgmFMError.h"
#include "DcgmFMCommon.h"
#include "fabricmanager.pb.h"
#include "DcgmLocalFabricManager.h"
#include "DcgmFMCommCtrl.h"
#include "lwml.h"
#include "lwml_internal.h"
#include "workqueue.h"
#include <g_lwconfig.h>


class DcgmLocalControlMsgHndl : public FMMessageHandler
{
public:
    DcgmLocalControlMsgHndl(DcgmLocalFabricManagerControl *pControl,
                            etblLWMLCommonInternal_st * etblLwmlCommonInternal);

    virtual ~DcgmLocalControlMsgHndl();

    virtual void handleEvent( FabricManagerCommEventType eventType, uint32 nodeId );

    void virtual handleMessage( lwswitch::fmMessage *pFmMessage );
    void virtual dumpMessage( lwswitch::fmMessage *pFmMessage );

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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void handlePortRmapConfigReqMsg( lwswitch::fmMessage *pFmMessage );
    void handlePortRidConfigReqMsg( lwswitch::fmMessage *pFmMessage );
    void handlePortRlanConfigReqMsg( lwswitch::fmMessage *pFmMessage );
#endif

    FM_ERROR_CODE SendMessageToGfm( lwswitch::fmMessage *pFmMessage, bool trackReq );

    DcgmLocalFabricManagerControl           *mpControl;        // server connection from lfm to gfm
    etblLWMLCommonInternal_st * metblLwmlCommonInternal;       // lwml internal function table 
    std::vector <workqueue_t *> mvWorkqueue;                   // work queues for processing message
};

/**
 * Structure for FM message user data to push to worker queue
 */
typedef struct FmMessageUserData
{
    lwswitch::fmMessage     *pFmMessage; /* Represents FM message  */
    DcgmLocalControlMsgHndl *pHndl;      /* Represents DcgmLocalControlMsgHndl object */
} FmMessageUserData_t;

#endif
