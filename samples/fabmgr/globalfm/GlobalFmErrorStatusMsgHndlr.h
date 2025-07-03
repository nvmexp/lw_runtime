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

#include <iostream>
#include <fstream>
#include <string>
#include <google/protobuf/text_format.h>
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"
#include "fabricmanager.pb.h"
#include "GlobalFabricManager.h"
#include "GlobalFmDegradedModeMgr.h"
#include "FMCommCtrl.h"

/*********************************************************************************/
/* Global FM msg handler for handling error reporting information from Local FMs */
/*********************************************************************************/


class GlobalFmErrorStatusMsgHndlr : public FMMessageHandler
{
public:
    GlobalFmErrorStatusMsgHndlr(GlobalFabricManager *pGfm);
    virtual ~GlobalFmErrorStatusMsgHndlr();

    virtual void handleEvent(FabricManagerCommEventType eventType, uint32 nodeId);
    void virtual handleMessage(lwswitch::fmMessage *pFmMessage);
    void virtual dumpMessage(lwswitch::fmMessage *pFmMessage);

private:
    void handleLWSwitchFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage);

    void handleLWSwitchFatalErrorScopeMsg(lwswitch::fmMessage *pFmMessage);

    void handleLWSwitchNonFatalErrorReportMsg(lwswitch::fmMessage *pFmMessage);

    void handleLWLinkErrorRecoveryMsg(lwswitch::fmMessage *pFmMessage);

    void handleGpuLWLinkFatalErrorMsg(lwswitch::fmMessage *pFmMessage);

    void parseLWSwitchErrorReportMsg(const lwswitch::switchErrorReport &errorReportMsg,
                                     uint32 nodeId);

    void logLWSwitchError(uint32_t switchPhysicalId,
                          const lwswitch::switchErrorInfo &errorInfo,
                          uint32 nodeId);

    FMIntReturn_t sendMessage(uint32 nodeId, lwswitch::fmMessage *pFmMessage);

    GlobalFabricManager *mpGfm;
};

