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

/*****************************************************************************/
/*  Abstract all the globalFM error handling logic                           */
/*****************************************************************************/

#include "fabricmanager.pb.h"
#include "FMCommonTypes.h"

class GlobalFabricManager;

typedef enum {
    ERROR_SOURCE_GPU            = 1,
    ERROR_SOURCE_LWSWITCH       = 2,
    ERROR_SORUCE_SW_LOCALFM     = 3,
    ERROR_SOURCE_SW_GLOBALFM    = 4,
    ERROR_SOURCE_MAX            = 5 // should be the last value
} GlobalFMErrorSource;

typedef enum {
    ERROR_TYPE_LWLINK_RECOVERY                 = 1,
    ERROR_TYPE_LWLINK_FATAL                    = 2,
    ERROR_TYPE_FATAL                           = 3, // general fatal error on GPU/LWSwtich
    ERROR_TYPE_CONFIG_NODE_FAILED              = 4,
    ERROR_TYPE_CONFIG_SWITCH_FAILED            = 5,
    ERROR_TYPE_CONFIG_SWITCH_PORT_FAILED       = 6,
    ERROR_TYPE_CONFIG_GPU_FAILED               = 7,
    ERROR_TYPE_CONFIG_TIMEOUT                  = 8,
    ERROR_TYPE_SOCKET_DISCONNECTED             = 9,
    ERROR_TYPE_HEARTBEAT_FAILED                = 10,
    ERROR_TYPE_LWSWITCH_FATAL_SCOPE            = 11, // LWSwitch fatal scope
    ERROR_TYPE_MAX                                  // should be the last value.
} GlobalFMErrorTypes;

class GlobalFMErrorHndlr
{
public:
    GlobalFMErrorHndlr(GlobalFabricManager *pGfm,
                       uint32_t nodeId,
                       uint32_t partitionId,
                       GlobalFMErrorSource errSource,
                       GlobalFMErrorTypes  errType,
                       lwswitch::fmMessage &errMsg);

    ~GlobalFMErrorHndlr();

    void processErrorMsg(void);

private:
    void handleErrorLWLinkRecovery(void);
    void handleErrorLWLinkFatal(void);
    void handleErrorSwitchFatalScope(void);
    void handleErrorConfigFailed(void);
    void handleErrorSocketDisconnected(void);
    void handleErrorHeartbeatFailed(void);
    bool isLWSwitchFatalErrorOnInactiveLWLinks(uint32_t nodeId, uint32_t physicalId,
                                               std::list<uint32_t> &portList);
    bool isLWSwitchFatalErrorOnAccessLWLinks(uint32_t nodeId, uint32_t physicalId,
                                             std::list<uint32_t> &portList,
                                             std::set<uint32_t> &gpusToReset);

    typedef struct {
        uint32_t nodeId;
        uint32_t partitionId;
        GlobalFMErrorSource errSource;
        GlobalFMErrorTypes  errType;
        lwswitch::fmMessage errMsg;
    } GlobalFMErrorInfo;

    GlobalFabricManager *mGfm;
    GlobalFMErrorInfo mErrorInfo;
};
