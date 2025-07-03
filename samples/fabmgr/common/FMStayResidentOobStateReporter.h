/*
 *  Copyright 2019-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

/*
    This class, which opens the switch devices,
    has been created for the purpose of reporting the error state
    to the driver when either the LFM or GFM throw errors and FM is in the 
    continue with failure state. An object to this class is created only
    when there is an error thrown either from lfm or gfm and if the config option
    FM_STAY_RESIDENT_ON_FAILURES=1. In that case, this class opens the switch 
    interfaces, and then reports LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_ERROR on the
    devices. Must revisit for MultiNode as this approach will not work in that case.

    Most of these functions are from the LocalFmSwitchIntf.h/cpp. If any changes are
    made to that file, make sure that such updates are made to this file as well
*/

#pragma once

#include <queue>
#include <stdio.h>
#include <fcntl.h>
extern "C"
{
    #include "lwswitch_user_api.h"
    #include "lwlink_user_api.h"
}

#include "ioctl_dev_lwswitch.h"
#include "FMErrorCodesInternal.h"
#include "FMCommonTypes.h"


class FMStayResidentOobStateSwitchIntf;

class FMStayResidentOobStateReporter
{
public:
    FMStayResidentOobStateReporter();
    ~FMStayResidentOobStateReporter();

    void reportFabricManagerStayResidentError(void);
    void setFmDriverStateToStandby();

private:
    void openLWSwitchDevices(void);
    void closeLWSwitchDevices(void);
    bool setFabricManagerDriverState(LWSWITCH_DRIVER_FABRIC_STATE driverState);
    std::vector <FMStayResidentOobStateSwitchIntf *> mvSwitchInterface;
};

class FMStayResidentOobStateSwitchIntf
{
    friend class FMStayResidentOobStateReporter;

public:
    FMStayResidentOobStateSwitchIntf(LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo);
    ~FMStayResidentOobStateSwitchIntf();

private:

    FMIntReturn_t setFmDriverState(LWSWITCH_DRIVER_FABRIC_STATE driverState);

    typedef struct
    {
        int    type;
        void  *ioctlParams;
        int    paramSize;
    } switchIoctl;

    typedef switchIoctl ioctl_t;

    lwswitch_device *mpLWSwitchDev;
    void acquireFabricManagementCapability();
    FMIntReturn_t doIoctl( ioctl_t *pIoctl );

    uint32_t mPhysicalId;
    uint32_t mSwitchInstanceId;
    FMPciInfo_t mPciInfo;
};
