/*
 *  Copyright 2018-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#pragma once
 
#include <queue>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

extern "C"
{
    #include "lwswitch_user_api.h"
}

#include "ioctl_dev_lwswitch.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"

/*****************************************************************************/
/*  This is a thread that gets spawned per-switch by Local Fabric Manager.   */      
/*  It operates on a vector of IOCTL pointers, which it issues serially to   */
/*  its instance of the switch driver.                                       */
/*****************************************************************************/

typedef enum SwitchIoctlStatus_enum
{
    SWITCH_IOCTL_SUCCESS = 0,
    SWITCH_IOCTL_FAIL    = 1
} SwitchIoctlStatus_t;

typedef struct
{
    int    type;
    void  *ioctlParams;
    int    paramSize;
} switchIoctl_struct;

typedef switchIoctl_struct switchIoctl_t;

typedef struct
{
    uint32_t          switchPhysicalId;
    LWSWITCH_ERROR    switchError;
} SwitchError_struct;

typedef struct
{
    uint32_t                       switchPhysicalId;
    LWSWITCH_GET_INTERNAL_LATENCY  latencies;
} SwitchLatency_struct;

typedef struct
{
    uint32_t                        switchPhysicalId;
    LWSWITCH_GET_LWLIPT_COUNTERS    counters;
} LwlinkCounter_struct;

/*****************************************************************************/

class LocalFMSwitchEventReader;

class LocalFMSwitchInterface
{
    friend class LocalFMSwitchEventReader;

public:
    LocalFMSwitchInterface( LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo,
                            uint32_t switchHeartbeatTimeout );
    ~LocalFMSwitchInterface();

    FMIntReturn_t doIoctl( switchIoctl_t *ioctl );
    uint32_t getSwitchDevIndex();
    uint32_t getSwitchPhysicalId();
    const FMPciInfo_t& getSwtichPciInfo();
    uint32_t getNumPorts();
    uint64_t getEnabledPortMask();
    uint32_t getSwitchArchType();
    const FMUuid_t& getUuid();
    void updateEnabledPortMask();

    FMIntReturn_t getSwitchErrors(LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                                  LWSWITCH_GET_ERRORS_PARAMS &ioctlParams);

private:
    void dumpIoctl( switchIoctl_t *ioctl );

    void dumpIoctlGetInfo( switchIoctl_t *ioctl );
    void dumpIoctlSwitchConfig( switchIoctl_t *ioctl );
    void dumpIoctlSwitchPortConfig( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressReq( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressReq( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressValid( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressResp( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressResp( switchIoctl_t *ioctl );
    void dumpIoctlSetGangedLink( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressReqLinkID( switchIoctl_t *ioctl );

    void dumpIoctlSetLatencyBin( switchIoctl_t *ioctl );
    void dumpIoctlGetLwspliptCounterConfig( switchIoctl_t *ioctl );
    void dumpIoctlSetLwspliptCounterConfig( switchIoctl_t *ioctl );

    void dumpIoctlGetErrors( switchIoctl_t *ioctl );
    void dumpIoctlGetFatalErrorScope( switchIoctl_t *ioctl );
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    void dumpIoctlGetFomValues( switchIoctl_t *ioctl );
    void dumpIoctlGetGradingValues( switchIoctl_t *ioctl );
#endif
    void dumpIoctlUnregisterLink( switchIoctl_t *ioctl );
    void dumpIoctlResetAndDrainLinks( switchIoctl_t *ioctl );

    void dumpIoctlExcludedDevice( switchIoctl_t *ioctl );
    void dumpIoctlSetFmDriverFabricState( switchIoctl_t *ioctl );
    void dumpIoctlSetFmDeviceFabricState( switchIoctl_t *ioctl );
    void dumpIoctlSetFmHeartbeatTimeout( switchIoctl_t *ioctl );

    void dumpIoctlSetRemapPolicy( switchIoctl_t *ioctl );
    void dumpIoctlSetRoutingId( switchIoctl_t *ioctl );
    void dumpIoctlSetRoutingLan( switchIoctl_t *ioctl );

    void dumpIoctlGetRemapPolicy( switchIoctl_t *ioctl );
    void dumpIoctlGetRoutingId( switchIoctl_t *ioctl );
    void dumpIoctlGetRoutingLan( switchIoctl_t *ioctl );

    void dumpIoctlGetLwliptCounters( switchIoctl_t *ioctl );
    void dumpIoctlGetThrouputCounters( switchIoctl_t *ioctl );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    void dumpIoctlSetMcRidTable( switchIoctl_t *ioctl );
    void dumpIoctlGetMcRidTable( switchIoctl_t *ioctl );
#endif

    void dumpIoctlTrainingErrorInfo( switchIoctl_t *ioctl );

    bool fetchSwitchPhysicalId();
    bool fetchSwitchPortInfo();
    bool fetchSwitchArchInfo();
    void acquireFabricManagementCapability();
    bool setSwitchHeartbeatKeepAliveTimeout(uint32_t switchHeartbeatTimeout);

    uint64_t getErrorIndex(LWSWITCH_ERROR_SEVERITY_TYPE errorType);
    void setErrorIndex(LWSWITCH_ERROR_SEVERITY_TYPE errorType, uint64_t errorIndex);
    void flushSwitchError(LWSWITCH_ERROR_SEVERITY_TYPE errorType);

    lwswitch_device *mpLWSwitchDev;
    uint32_t mPhysicalId;
    uint32_t mSwitchInstanceId;
    FMPciInfo_t mPciInfo;
    uint32_t mNumPorts;
    uint32_t mNumVCs;
    uint64_t mEnabledPortMask;
    uint32_t mArchType;
    FMUuid_t mUuid;

    uint64_t mSwitchErrorIndex[LWSWITCH_ERROR_SEVERITY_MAX];
};

