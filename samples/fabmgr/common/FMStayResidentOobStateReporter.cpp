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
#include <stdio.h>
#include <queue>
#include <stdio.h>
#include <fcntl.h>
#include <sstream>
#include <stdexcept>

#include "fm_log.h"
#include "FMGpuDriverVersionCheck.h"
#include "FMStayResidentOobStateReporter.h"

FMStayResidentOobStateReporter::FMStayResidentOobStateReporter()
{
    mvSwitchInterface.clear();

    //
    // before accessing any Driver, validate the version. This will throw
    // exception if the version doesn't match or whitelisted
    //
    FMGpuDriverVersionCheck gpuDrvVersionChk;
    gpuDrvVersionChk.checkGpuDriverVersionCompatibility("fabric manager");

    // driver version is compatible, open LWSwitch devices
    openLWSwitchDevices();
}

FMStayResidentOobStateReporter::~FMStayResidentOobStateReporter()
{
    // close all the devices
    closeLWSwitchDevices();
    mvSwitchInterface.clear();
}

void
FMStayResidentOobStateReporter::reportFabricManagerStayResidentError(void)
{
    // set the driver fabric manager state to MANAGER_ERROR
    bool retVal = setFabricManagerDriverState(LWSWITCH_DRIVER_FABRIC_STATE_MANAGER_ERROR);

    // treat error as exception as we need all the switch to report uniform manager error state
    if (retVal == false) {
        std::ostringstream ss;
        ss << "request to set LWSwitch driver fabric manager state information failed";
        throw std::runtime_error(ss.str());
    }
}

void
FMStayResidentOobStateReporter::setFmDriverStateToStandby(void)
{
    // set the driver fabric manager state to STATE_STANDBY
    bool retVal = setFabricManagerDriverState(LWSWITCH_DRIVER_FABRIC_STATE_STANDBY);

    // log failure as error as FM is exiting and best case. Detailed IOCTL failure is already logged.
    if (retVal == false) {
        FM_LOG_ERROR("request to set fabric state to standby during fabric manager shutdown failed for some LWSwitches");
    }
}

void
FMStayResidentOobStateReporter::openLWSwitchDevices(void)
{
    // get switch instances and start the corresponding switch interface
    LWSWITCH_GET_DEVICES_V2_PARAMS params;
    LW_STATUS status;
    uint32_t i;

    status = lwswitch_api_get_devices(&params);

    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            std::ostringstream ss;
            ss << "fabric manager version is incompatible with LWSwitch driver. Please update with matching LWPU driver package";
            FM_LOG_ERROR("%s", ss.str().c_str());
            throw std::runtime_error(ss.str());
        }
        // all other errors, log the error code and bail out
        std::ostringstream ss;
        ss << "request to query LWSwitch device information from LWSwitch driver failed with error:" << lwstatusToString(status);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    for (i = 0; i < params.deviceCount; i++)
    {
        FMStayResidentOobStateSwitchIntf *pSwitchDevice;
        // ignore excluded switches
        if (params.info[i].deviceReason != LWSWITCH_DEVICE_BLACKLIST_REASON_NONE) {
            continue;
        }

        pSwitchDevice = new FMStayResidentOobStateSwitchIntf( params.info[i] );
        mvSwitchInterface.push_back( pSwitchDevice );
    }
}

void
FMStayResidentOobStateReporter::closeLWSwitchDevices(void)
{
    std::vector <FMStayResidentOobStateSwitchIntf *>::iterator it = mvSwitchInterface.begin();
    while (it != mvSwitchInterface.end()) {
        FMStayResidentOobStateSwitchIntf *pSwitchDevice = *it;
        it = mvSwitchInterface.erase(it);
        delete pSwitchDevice;
    }
}

bool FMStayResidentOobStateReporter::setFabricManagerDriverState(LWSWITCH_DRIVER_FABRIC_STATE driverState)
{
    FMIntReturn_t rc;
    int retVal = true;

    // for each device, set the fm driver state
    std::vector <FMStayResidentOobStateSwitchIntf *>::iterator it;
    for (it = mvSwitchInterface.begin(); it != mvSwitchInterface.end(); it++) {
        FMStayResidentOobStateSwitchIntf *pSwitchIntf = *(it);
        rc = pSwitchIntf->setFmDriverState(driverState);
        if (rc != FM_INT_ST_OK) {
            // continue to next switch, but keep the over-all status as failure
            retVal = false;
        }
    }

    return retVal;
}

FMStayResidentOobStateSwitchIntf::FMStayResidentOobStateSwitchIntf(LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo)
{
    LW_STATUS retVal;

    mpLWSwitchDev = NULL;

    // update our local PCI BDF information
    mPciInfo.domain = switchInfo.pciDomain;
    mPciInfo.bus = switchInfo.pciBus;
    mPciInfo.device = switchInfo.pciDevice;
    mPciInfo.function = switchInfo.pciFunction;
    snprintf(mPciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
             FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&mPciInfo));

    // cache switch instance/enumeration index
    mSwitchInstanceId = switchInfo.deviceInstance;
    
    retVal = lwswitch_api_create_device(&switchInfo.uuid, &mpLWSwitchDev);
    if ( retVal != LW_OK )
    {
        std::ostringstream ss;
        ss << "request to open handle to LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId
           << " failed with error:" << lwstatusToString(retVal);
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    //
    // before issuing any IOCTL to the driver, acquire our fabric management capability which indicate
    // fabric manager has enough permission to issue privileged IOCTLs. This is required to run FM
    // from root and non-root context. Also, capability is per switch device
    //

    // this function will throw exception if we fail to acquire the required capability
    acquireFabricManagementCapability();
}

FMStayResidentOobStateSwitchIntf::~FMStayResidentOobStateSwitchIntf()
{
    if ( mpLWSwitchDev )
    {
        lwswitch_api_free_device(&mpLWSwitchDev);
        mpLWSwitchDev = NULL;
    }
}

void FMStayResidentOobStateSwitchIntf::acquireFabricManagementCapability()
{
    LW_STATUS retVal;

    //
    // by default all the switch device nodes (/dev/lwpu-lwswitchX in Linux ) has permission
    // to all users. All the privileged access is controlled through special fabric management node.
    // The actual management node dependents on driver mechanism. For devfs based support, it will be
    // /dev/lwpu-caps/lwpu-capX and for procfs based, it will be /proc/driver/lwpu-lwlink/capabilities/fabric-mgmt
    // This entry is created by driver and default access is for root/admin. The system administrator then must
    // change access to desired user. The below API is verifying whether FM has access to the path, if so open it
    // and associate/link the corresponding file descriptor with the file descriptor associated with the
    // current device node file descriptor (ie fd of /dev/lwpu-lwswitchX)
    //

    retVal = lwswitch_api_acquire_capability(mpLWSwitchDev, LWSWITCH_CAP_FABRIC_MANAGEMENT);
    if ( retVal != LW_OK )
    {
        // failed to get capability. throw error based on common return values
        switch (retVal)
        {
            case LW_ERR_INSUFFICIENT_PERMISSIONS:
            {
                std::ostringstream ss;
                ss << "failed to acquire required privileges to access LWSwitch devices." <<
                      " make sure fabric manager has access permissions to required device node files";
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
                break;
            }
            case LW_ERR_NOT_SUPPORTED:
            {
                //
                // driver doesn't have fabric management capability support on Windows for now and will 
                // return LW_ERR_NOT_SUPPORTED. So, treat this as not an error for now to let Windows FM to
                // continue. In Linux, the assumption is that Driver will not return LW_ERR_NOT_SUPPORTED
                // and even if, FM will eventually fail as privileged control calls will start erroring out.
                //
                break;
            }
            default:
            {
                std::ostringstream ss;
                ss << "request to acquire required privileges to access LWSwitch devices failed with error:" << lwstatusToString(retVal);
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
                break;
            }
        }
    }

    // successfully acquired required fabric management capability
}

FMIntReturn_t
FMStayResidentOobStateSwitchIntf::doIoctl( ioctl_t *pIoctl )
{
    LW_STATUS retVal;

    switch ( pIoctl->type )
    {
        // only IOCTL_LWSWITCH_SET_FM_DRIVER_STATE is needed and supported in this class
        case IOCTL_LWSWITCH_SET_FM_DRIVER_STATE:
            break;

        default:
            FM_LOG_ERROR("unsupported ioctl type passed to switch interface for out-of-band error reporting");
            return FM_INT_ST_IOCTL_ERR;
    }

    retVal = lwswitch_api_control( mpLWSwitchDev, pIoctl->type, pIoctl->ioctlParams, pIoctl->paramSize );

    if ( retVal != LW_OK ) {
        
        FM_LOG_ERROR( "LWSwitch driver ioctl type 0x%x failed for device index %d physical id %d pci bus id %s with error %s",
                      pIoctl->type, mSwitchInstanceId, mPhysicalId, mPciInfo.busId, lwstatusToString(retVal) );
        return FM_INT_ST_IOCTL_ERR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t 
FMStayResidentOobStateSwitchIntf::setFmDriverState(LWSWITCH_DRIVER_FABRIC_STATE driverState)
{
    ioctl_t ioctlStruct;
    LWSWITCH_SET_FM_DRIVER_STATE_PARAMS ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));

    ioctlParams.driverState = driverState;
    ioctlStruct.type = IOCTL_LWSWITCH_SET_FM_DRIVER_STATE;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    return doIoctl( &ioctlStruct );
}
