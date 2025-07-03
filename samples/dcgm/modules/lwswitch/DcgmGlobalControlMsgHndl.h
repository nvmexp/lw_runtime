#ifndef DCGM_GLOBAL_CONTROL_MSG_HNDL_H
#define DCGM_GLOBAL_CONTROL_MSG_HNDL_H

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "DcgmFMCommon.h"
#include "fabricmanager.pb.h"
#include "DcgmFMCommCtrl.h"

class DcgmGlobalControlMsgHndl : public FMMessageHandler
{
public:
    DcgmGlobalControlMsgHndl( FMConnInterface *ctrlConnIntf );

    virtual ~DcgmGlobalControlMsgHndl();

    virtual void handleEvent( FabricManagerCommEventType eventType, uint32 nodeId );

    void virtual handleMessage( lwswitch::fmMessage *pFmMessage);

    void virtual dumpMessage( lwswitch::fmMessage *pFmMessage );

private:
    void handleNodeStatsAckMsg( lwswitch::fmMessage *pFmMessage );

    FMConnInterface *mCtrlConnIntf;
};

#endif
