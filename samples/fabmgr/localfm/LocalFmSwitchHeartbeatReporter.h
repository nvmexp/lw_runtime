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

#pragma once

#include "FMCommonTypes.h"
#include "fm_log.h"
#include "FMTimer.h"
#include "LocalFabricManager.h"

//
// we divide the total heartbeat timeout by this frequency. Then the timer is
// configured to expire in that interval.
//
#define FM_SWITCH_HEARTBEAT_FREQ  3

class LocalFmSwitchHeartbeatReporter
{
public:
	LocalFmSwitchHeartbeatReporter(LocalFabricManagerControl *localFm,
                                   unsigned int heartbeatTimeout);
	~LocalFmSwitchHeartbeatReporter();
	void startHeartbeatReporting();
	void stopHeartbeatReporting();

private:
	static void switchHeartbeatTimerCallback(void *ctx);
	FMIntReturn_t setDriverFabricManagerState(LWSWITCH_DRIVER_FABRIC_STATE driverFabricState);
	void setDeviceFabricManagerState(LWSWITCH_DEVICE_FABRIC_STATE deviceFabricState);
	void sendHeartbeatToDevices();

	LocalFabricManagerControl *mpLocalFm;
	FMTimer *mTimer;
	unsigned int mHeartbeatInterval; //in milliseconds
};
