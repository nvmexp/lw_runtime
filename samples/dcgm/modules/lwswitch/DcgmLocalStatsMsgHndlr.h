#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "DcgmFMCommon.h"
#include "fabricmanager.pb.h"
#include "DcgmFMCommCtrl.h"
#include "DcgmLocalStatsReporter.h"

/*******************************************************************************/
/* Local FM msg handler for on demand stats collection request from Global FM  */
/*******************************************************************************/

class DcgmLocalStatsReporter;

class DcgmLocalStatsMsgHndlr : public FMMessageHandler
{
public:
    DcgmLocalStatsMsgHndlr(FMConnInterface *ctrlConnIntf,
                           DcgmLocalStatsReporter *pLocalStatsReporter);

    virtual ~DcgmLocalStatsMsgHndlr();

    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);

    void virtual handleMessage(lwswitch::fmMessage *pFmMessage);
    void virtual dumpMessage(lwswitch::fmMessage *pFmMessage);

    void handleGetErrorRequestMsg(lwswitch::fmMessage *pFmMessage);

    void handleGetStatsRequestMsg(lwswitch::fmMessage *pFmMessage);

private:
    void handleFatalErrorReportAckMsg(lwswitch::fmMessage *pFmMessage);

    void handleNonFatalErrorReportAckMsg(lwswitch::fmMessage *pFmMessage);

    void handleNodeStatsAckMsg(lwswitch::fmMessage *pFmMessage);

    DcgmLocalStatsReporter *mLocalStatsReporter;
    FMConnInterface *mCtrlConnIntf;
};
