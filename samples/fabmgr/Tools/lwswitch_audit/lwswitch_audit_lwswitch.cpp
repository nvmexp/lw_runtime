#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include "lwos.h"
#include "logging.h"

#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"

#include "lwswitch_audit_lwswitch.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_node.h"

extern "C"
{
#include "lwswitch_user_api.h"
#include "lwlink_user_api.h"
}

#define UUID_BUFFER_SIZE 80 // ref LWML

void
lwswitch::acquireFabricManagementCapability(lwswitch_device *pLWSwitchDev)
{
    LW_STATUS retVal;

    //
    // by default all the switch device nodes (/dev/lwpu-lwswitchX in Linux ) grant permissions
    // to all users. All the privileged access is controlled through special fabric management node.
    // The actual management node dependents on driver mechanism. For devfs based support, it will be
    // /dev/lwpu-caps/lwpu-capX and for procfs based, it will be /proc/driver/lwpu-lwlink/capabilities/fabric-mgmt
    // This entry is created by driver and default access is for root/admin. The system administrator then must
    // change access to desired user. The below API is verifying whether FM has access to the path, if so open it
    // and associate/link the corresponding file descriptor with the file descriptor associated with the
    // current device node file descriptor (ie fd of /dev/lwpu-lwswitchX)
    //

    retVal = lwswitch_api_acquire_capability(pLWSwitchDev, LWSWITCH_CAP_FABRIC_MANAGEMENT);
    if ( retVal != LW_OK )
    {
        // failed to get capability. throw error based on common return values
        switch (retVal)
        {
            case LW_ERR_INSUFFICIENT_PERMISSIONS:
            {
                fprintf(stderr, "failed to acquire required privileges to access LWSwitch devices."
                        " make sure lwswitch-audit has access permissions to required device node files\n");
                exit ( -1 );
            }
            case LW_ERR_NOT_SUPPORTED:
            {
                //
                // driver doesn't have fabric management capability support on Windows for now and will
                // return LW_ERR_NOT_SUPPORTED. So, treat this as not an error for now to let Windows lwswitch-audit to
                // continue. In Linux, the assumption is that Driver will not return LW_ERR_NOT_SUPPORTED
                // and if it does lwswitch-audit will eventually fail as privileged control calls will start erroring out.
                //
                break;
            }
            default:
            {
                fprintf(stderr, "request to acquire required privileges to access LWSwitch devices failed with error:%s\n", lwstatusToString(retVal));
                exit ( -1 );
                break;
            }
        }
    }
    // successfully acquired required fabric management capability
}


/*
Opens lwswitch device
switchInfo: Switch info for device to be opened
Return: Pointer to opened device. This pointer is used to call lwswitch_api_* functions
*/
lwswitch_device *
lwswitch::openSwitchDev(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo)
{
    lwswitch_device *pLWSwitchDev;
    pLWSwitchDev = NULL;

    if (switchInfo.deviceReason != LWSWITCH_DEVICE_BLACKLIST_REASON_NONE) {
        // the switch is excluded, do not try to open
        // this function can be called from getArch which tried to open switch to get arch.
        return NULL;
    }

    // update local uuid information. first clear our entire UUID memory area and copy only what shim layer returns.
    char uuidBuf[UUID_BUFFER_SIZE];
    memset(uuidBuf, 0, UUID_BUFFER_SIZE);
    lwswitch_uuid_to_string(&switchInfo.uuid, uuidBuf, UUID_BUFFER_SIZE);

    LW_STATUS retVal = lwswitch_api_create_device(&switchInfo.uuid, &pLWSwitchDev);
    if ( retVal != LW_OK )
    {
        // the tool opens lwswitch at the beginning to get platform arch
        // this could fail because the switch is excluded or degraded
        PRINT_VERBOSE("request to open handle to LWSwitch index: %d pci bus id: %d failed with error: %s\n",
                      switchInfo.deviceInstance, switchInfo.pciBus, lwstatusToString(retVal));
        return NULL;
    }

    //
    // before issuing any IOCTL to the driver, acquire our fabric management capability which indicates that
    // lwswitch-audit has permission to issue privileged IOCTLs. This is required to run lwswitch-audit
    // from root and non-root context. Also, capability is per switch device
    //
    lwswitch::acquireFabricManagementCapability( pLWSwitchDev );
    return pLWSwitchDev;
}

/*
Get switch physical id by reading the LWSwitch
Return: switch physical id
*/
int
lwswitch::readPhyId()
{
    uint32_t phyId = -1;
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;

    LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_INFO, &ioctlParams, sizeof(ioctlParams) );
    if ( retVal != LW_OK )
    {
        fprintf(stderr,  "request to read switch physical ID for device index %d physical id %d pci bus id %d failed with error %s\n",
                mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
    }
    else
    {
        phyId = ioctlParams.info[0];
    }
    return phyId;
}

lwswitch::lwswitch(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &devInfo)
{
    // update our local PCI BDF information
    mPciInfo.domain = devInfo.pciDomain;
    mPciInfo.bus = devInfo.pciBus;
    mPciInfo.device = devInfo.pciDevice;
    mPciInfo.function = devInfo.pciFunction;
    
    // cache switch instance/enumeration index
    mDevId = devInfo.deviceInstance;

    mpLWSwitchDev = openSwitchDev(devInfo);

    //Read arch type of switch
    if ( !( mArchType = lwswitch::getSwitchArchInfo( mpLWSwitchDev ) ))
    {
        fprintf(stderr,  "failed to read architecture type for device index %d pci bus id %d\n",
                mDevId, mPciInfo.bus );
        exit( -1 );
    }

    // update our gpio based switch physical id
    if ( (mPhyId = readPhyId()) == -1 )
    {
        fprintf(stderr, "failed to get GPIO based physical id for LWSwitch index: %d pci bus id: %d\n",
                mDevId, mPciInfo.bus);
        exit( -1 );
    }
}

lwswitch::~lwswitch()
{
    if ( mpLWSwitchDev )
    {
        lwswitch_api_free_device(&mpLWSwitchDev);
        mpLWSwitchDev = NULL;
    }
}

/*Get port mask for lwswitch device
  portMask: reference variable in which value of 64-bit mask is returned
  Return: true = successfully read mask
          false = failed to read mask
*/
bool
lwswitch::readPortMask(uint64_t &portMask)
{
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 2;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0;
    ioctlParams.index[1] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32;
    
    LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_INFO, &ioctlParams, sizeof(ioctlParams) );

    if ( retVal != LW_OK )
    {
        fprintf(stderr, "request to read port mask failed for device index %d physical id %d pci bus id %d with error %s\n",
                 mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
        return false;
    }
    else
    {
        portMask = ((uint64_t)ioctlParams.info[1] << 32) | (uint64_t)ioctlParams.info[0];
        return true;
    }
}

/*Is port enabled
mask: Port mask previously read from device
portNum: port number to check if enabled disabled
Return: true=enabled, false=disabled
*/
bool
lwswitch::isPortEnabled(uint64_t mask, int portNum)
{
    if((portNum > 63) || (portNum < 0))
    {
        fprintf(stderr, "naIsPortEnabled: port number %d invalid\n", portNum);
        return false;
    }
    if((mask >> portNum) & 0x1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/*Read requestor link ID from switch
switchPort: port from which to read the requestor link ID
Return: requestor link ID
*/
int
lwswitch::readReqLinkId( uint32_t switchPort)
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS idParams;
    idParams.portNum = switchPort;

    LW_STATUS retVal = lwswitch_api_control( mpLWSwitchDev, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &idParams, sizeof(idParams) );
    if ( retVal != LW_OK )
    {
        fprintf(stderr, "request to read Requestor link ID for device index %d physical id %d pci bus id %d failed with error %s\n",
                mDevId, mPhyId, mPciInfo.bus, lwstatusToString(retVal));
        return -1;
    }
    else
        return idParams.requesterLinkID;
}

unsigned int
lwswitch::getSwitchArchInfo( lwswitch_device * pLWSwitchDev)
{
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;
    ioctlParams.count = 1;

    LW_STATUS retVal = lwswitch_api_control( pLWSwitchDev, IOCTL_LWSWITCH_GET_INFO, &ioctlParams, sizeof(ioctlParams) );
    if ( retVal != LW_OK )
    {
        return 0;
    }
    return ioctlParams.info[0];
}
