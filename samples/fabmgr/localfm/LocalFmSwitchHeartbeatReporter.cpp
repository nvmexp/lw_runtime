#include <stdio.h>
#include <iostream>

#include "LocalFmSwitchHeartbeatReporter.h"

LocalFmSwitchHeartbeatReporter::LocalFmSwitchHeartbeatReporter(LocalFabricManagerControl *localFm,
                                                               unsigned int heartbeatTimeout)
{
    mpLocalFm = localFm;
    mHeartbeatInterval = heartbeatTimeout / FM_SWITCH_HEARTBEAT_FREQ;   //this is in milliseconds
    // create a timer
    mTimer = new FMTimer(LocalFmSwitchHeartbeatReporter::switchHeartbeatTimerCallback, this);
}

LocalFmSwitchHeartbeatReporter::~LocalFmSwitchHeartbeatReporter()
{
    delete mTimer;
    mTimer = NULL;
}

void
LocalFmSwitchHeartbeatReporter::switchHeartbeatTimerCallback(void *ctx)
{
    LocalFmSwitchHeartbeatReporter *pSwitchHeartbeat = (LocalFmSwitchHeartbeatReporter*) ctx;
    pSwitchHeartbeat->sendHeartbeatToDevices();
}

void
LocalFmSwitchHeartbeatReporter::sendHeartbeatToDevices()
{
	FMIntReturn_t rc = FM_INT_ST_OK;

	rc = setDriverFabricManagerState(LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED);
	if (rc == FM_INT_ST_OK) {
		mTimer->restart();
	}
    else {
        std::ostringstream ss;
        ss << "failed to report heartbeat to LWSwitch driver, stopping further heartbeat reporting";
        FM_LOG_ERROR("%s", ss.str().c_str());
        FM_SYSLOG_ERR("%s", ss.str().c_str());
    }
}

void
LocalFmSwitchHeartbeatReporter::startHeartbeatReporting()
{
	// set switch devices state to configured
	setDeviceFabricManagerState(LWSWITCH_DEVICE_FABRIC_STATE_CONFIGURED);

    // set the FM state to toggle the state immediately.
    setDriverFabricManagerState(LWSWITCH_DRIVER_FABRIC_STATE_CONFIGURED);

	//start heartbeat timer to report driver state
	mTimer->start(mHeartbeatInterval/1000);
}

void
LocalFmSwitchHeartbeatReporter::stopHeartbeatReporting()
{
	mTimer->stop();
}

void 
LocalFmSwitchHeartbeatReporter::setDeviceFabricManagerState(LWSWITCH_DEVICE_FABRIC_STATE deviceFabricState)
{
    FMIntReturn_t rc;
    for (int i = 0; i < (int)mpLocalFm->getNumLwswitchInterface(); i++) {
        LocalFMSwitchInterface *pSwitchIntf = mpLocalFm->switchInterfaceAtIndex(i);

        if (pSwitchIntf == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object.");
            return;
        }

        // ignore degraded switches
        if (mpLocalFm->isSwitchDegraded(pSwitchIntf->getSwitchPhysicalId())) {
            continue;
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS ioctlParams;

        memset(&ioctlParams, 0, sizeof(ioctlParams));
        ioctlParams.deviceState = deviceFabricState;
        ioctlStruct.type = IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        rc = pSwitchIntf->doIoctl( &ioctlStruct );
        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to set LWSwitch device fabric state to configured state");
        }
    }
}

FMIntReturn_t
LocalFmSwitchHeartbeatReporter::setDriverFabricManagerState(LWSWITCH_DRIVER_FABRIC_STATE driverFabricState)
{
    FMIntReturn_t rc = FM_INT_ST_OK;

    for (int i = 0; i < (int)mpLocalFm->getNumLwswitchInterface(); i++) {
        LocalFMSwitchInterface *pSwitchIntf = mpLocalFm->switchInterfaceAtIndex(i);

        if (pSwitchIntf == NULL) {
            FM_LOG_ERROR("failed to get LWSwitch driver interface object.");
            return FM_INT_ST_BADPARAM;
        }

        // ignore degraded switches
        if (mpLocalFm->isSwitchDegraded(pSwitchIntf->getSwitchPhysicalId())) {
            continue;
        }

        switchIoctl_t ioctlStruct;
        LWSWITCH_SET_FM_DRIVER_STATE_PARAMS ioctlParams;

        memset(&ioctlParams, 0, sizeof(ioctlParams));
        ioctlParams.driverState = driverFabricState;
        ioctlStruct.type = IOCTL_LWSWITCH_SET_FM_DRIVER_STATE;
        ioctlStruct.ioctlParams = &ioctlParams;
        ioctlStruct.paramSize = sizeof(ioctlParams);

        rc = pSwitchIntf->doIoctl( &ioctlStruct );
        if (rc != FM_INT_ST_OK) {
            FM_LOG_ERROR("failed to set LWSwitch driver fabric manager state");
            return rc;
        }
    }

    return rc;
}
