#include <fcntl.h>
#include <limits.h>
#include <map>
#include <stdbool.h>
#include <stdio.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <syslog.h>
#include <set>
#include <unistd.h>

#include "lwos.h"
#include "lwswitch_audit_node.h"
#include "logging.h"
#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_dev.h"

//map switch instance /dev/lwpu-lwswitch<dev id> to switch_id (0-12 as used in HWLinks table)
static std::map<int, int> dev_id_to_sw_id;

//map switch_id (0-12 as used in HWLinks table) to switch instance /dev/lwpu-lwswitch<dev id>
static std::map<int, int> sw_id_to_dev_id;

/*Internal method
dev_id: Switch instance in /dev/
switch_id: Switch ID in HWLinks table
*/
static void setDevToSwitchMap(int dev_id, int switch_id)
{
    dev_id_to_sw_id[dev_id] = switch_id;
}

/*Internal method
dev_id: Switch instance in /dev/
switch_id: Switch ID in HWLinks table
*/
static void setSwitchToDevMap(int switch_id, int dev_id)
{
    sw_id_to_dev_id[switch_id] = dev_id;
}

/*
dev_id: Switch instance in /dev/
Return: Corresponding switch id in HWLinks table
*/
int naGetDevToSwitchID(int dev_id)
{
    std::map<int, int>::iterator it;
    if ( (it = dev_id_to_sw_id.find(dev_id)) != dev_id_to_sw_id.end() ) 
    {
        return it->second;
    } 
    else
    {
        return -1;
    }
}

/*
switch_id: Switch ID in HWLinks table
Return: Corresponding switch instance in /dev/
*/
int naGetSwitchToDevID(int switch_id)
{
    std::map<int, int>::iterator it;
    if ( (it = sw_id_to_dev_id.find(switch_id)) != sw_id_to_dev_id.end() )
    {
        return it->second;
    }
    else
    {
        return -1;
    }
}

/*
Opens lwswitch device
dev_id: Switch instance in /dev/
Return: File descriptor for opened device
*/
int naOpenSwitchDev(int dev_id)
{
    int fd;
    char switch_path[SWITCH_PATH_LEN];
    sprintf(switch_path, "/dev/lwpu-lwswitch%d", dev_id);
    if((fd = open(switch_path, O_WRONLY)) == -1) 
    {
        fprintf(stderr, "Unable to open device /dev/lwpu-lwswitch%d errno=%d", dev_id, errno);
        return -1;
    }
    else 
    {
        return fd;
    }

}

/*
Get switch physical id by reading the LWSwitch
dev_id: Device instance in /dev/
Return: switch physical id
*/
static int getSwitchPhyId(int fd)
{
    uint32_t phy_id = -1;
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;
    if(ioctl(fd, IOCTL_LWSWITCH_GET_INFO, &ioctlParams) == -1)
    {
        perror("Unable to read switch physical ID");
    }
    else
    {
        phy_id = ioctlParams.info[0];
    }
    return phy_id;
}

/*Get port Mast for lwswitch device
fd: file descriptor for opened switch device
mask: reference variable in which value of 64bitmask is returned
Return: true = successfully read mask
        false = failed to read mask
*/
bool naReadPortMask(int fd, uint64_t &port_mask)
{
    LWSWITCH_GET_INFO ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 2;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0;
    ioctlParams.index[1] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32;
    if(ioctl(fd, IOCTL_LWSWITCH_GET_INFO, &ioctlParams) == -1)
    {
        perror("Unable to read port-enabled mask for switch");
        return false;
    }
    else
    {
        port_mask = ((uint64_t)ioctlParams.info[1] << 32) | (uint64_t)ioctlParams.info[0];
        return true;
    }
}

/*Is port enabled
mask: Port mask previously read from device
port_num: port number to check if enabled disabled
Return: true=enabled, false=disabled
*/
bool naIsPortEnabled(uint64_t mask, int port_num)
{
    if((port_num > 63) || (port_num < 0))
    {
        fprintf(stderr, "naIsPortEnabled: port number %d invalid\n", port_num);
        return false;
    }
    if((mask >> port_num) & 0x1)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/*
Read switch physical IDs from switches and populate the 
switch ID <-> Switch instance maps
*/
bool naReadSwitchIDs(int num_switches)
{
    int fd;
    int dev_id=0;

    for(dev_id = 0; dev_id < num_switches; dev_id++) 
    {
        if((fd = naOpenSwitchDev(dev_id)) == -1)
        {
            return false;;
        }
        else
        {
            int phy_id;
            if((phy_id = getSwitchPhyId(fd)) == -1)
            {   
                close(fd);
                fprintf(stderr, "Unable to read physical id for /dev/lwpu-lwswitch%d\n", dev_id); 
                return false;
            }
            else
            {
                setDevToSwitchMap(dev_id, switchPhyIDtoSwitchID(phy_id));
                setSwitchToDevMap(switchPhyIDtoSwitchID(phy_id), dev_id);
                close(fd);
            }
        }
    }
    return true;
}

/*Read requestor link ID from switch
fd: File desciptor for switch
switch_port: port from which to read the requestor link ID
Return: requestor link ID
*/
int naReadReqLinkId(int fd, uint32_t switch_port)
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS id_params;

    id_params.portNum = switch_port;

    if(ioctl(fd, IOCTL_LWSWITCH_GET_INGRESS_REQLINKID, &id_params) == -1) 
        return -1;
    else
        return id_params.requesterLinkID;
}

