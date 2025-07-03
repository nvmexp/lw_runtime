#pragma once

/*****************************************************************************/
/*  Abstract all the globalFM error handling logic                           */
/*****************************************************************************/

#include "fabricmanager.pb.h"

class DcgmGlobalFabricManager;

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
    ERROR_TYPE_SHARED_PARTITION_CONFIG_FAILED  = 11,
    ERROR_TYPE_MAX                             = 12 // should be the last value.
} GlobalFMErrorTypes;

class DcgmGlobalFMErrorHndlr
{
public:
    DcgmGlobalFMErrorHndlr(DcgmGlobalFabricManager *pGfm,
                           uint32_t nodeId,
                           uint32_t partitionId,
                           GlobalFMErrorSource errSource,
                           GlobalFMErrorTypes  errType,
                           lwswitch::fmMessage &errMsg);

    ~DcgmGlobalFMErrorHndlr();

    void processErrorMsg(void);

private:
    void handleErrorLWLinkRecovery(void);
    void handleErrorLWLinkFatal(void);
    void handleErrorFatal(void);
    void handleErrorConfigFailed(void);
    void handleErrorSocketDisconnected(void);
    void handleErrorHeartbeatFailed(void);
    void handleErrorSharedPartitionConfigFailed(void);

    typedef struct {
        uint32_t nodeId;
        uint32_t partitionId;
        GlobalFMErrorSource errSource;
        GlobalFMErrorTypes  errType;
        lwswitch::fmMessage errMsg;
    } GlobalFMErrorInfo;

    DcgmGlobalFabricManager *mGfm;
    GlobalFMErrorInfo mErrorInfo;
};
