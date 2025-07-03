#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "LwcmCacheManager.h"
#include "dcgm_fields.h"
#include "DcgmFMError.h"
#include "DcgmFMCommon.h"
#include "fabricmanager.pb.h"
#include "DcgmGlobalFabricManager.h"
#include "DcgmFMCommCtrl.h"

/*******************************************************************************/
/* Global FM msg handler for stats collection information from Local FMs       */
/*******************************************************************************/


class DcgmGlobalStatsMsgHndlr : public FMMessageHandler
{
public:
    DcgmGlobalStatsMsgHndlr(DcgmGlobalFabricManager *pGfm);
    virtual ~DcgmGlobalStatsMsgHndlr();

    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);
    void virtual handleMessage(lwswitch::fmMessage *pFmMessage);
    void virtual dumpMessage(lwswitch::fmMessage *pFmMessage);

private:
    void parseStatsReportMsg(uint32 nodeId, const lwswitch::nodeStats &statsReportMsg);

    void parseErrorReportMsg(const lwswitch::switchErrorReport &errorReportMsg);
    
    void cacheLatency(PortLatencyHist_t &latencyHisto, uint32_t switchPhysicalId,
                      int portNum, int vcNum);

    void cacheCounters(LwlinkCounter_t &linkCounter, int switchPhysicalId, int portNum);

    void cacheDetailedSwitchError(uint32_t switchPhysicalId,
                                  const lwswitch::switchErrorInfo &errorInfo);

    void appendLWswitchSamples(dcgm_field_eid_t entityId, unsigned short dcgmFieldId,
                               dcgmcm_sample_p samples);

    void handleFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage);

    void handleNonFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage);

    void handleStatsReportMsg(lwswitch::fmMessage *pFmMessage);

    void handleNodeStatsAckMsg(lwswitch::fmMessage *pFmMessage);

    void handleLWLinkErrorRecoveryMsg(lwswitch::fmMessage *pFmMessage);

    void handleLWLinkErrorFatalMsg(lwswitch::fmMessage *pFmMessage);

    void logLWSwitchError(uint32_t switchPhysicalId, const lwswitch::switchErrorInfo &errorInfo);

    FM_ERROR_CODE sendMessage(uint32 nodeId, lwswitch::fmMessage *pFmMessage);

    DcgmGlobalFabricManager *mpGfm;
};

