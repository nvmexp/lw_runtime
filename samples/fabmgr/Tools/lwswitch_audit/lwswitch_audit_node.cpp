#include <errno.h>
#include <stdio.h>
#include <unistd.h>

#include "lwos.h"
#include "logging.h"

#include "ioctl_lwswitch.h"
#include "ioctl_dev_lwswitch.h"
#include "ioctl_common_lwswitch.h"

#include "lwswitch_audit_lwswitch.h"
#include "lwswitch_audit_logging.h"
#include "lwswitch_audit_node.h"

extern "C"
{
#include "lwswitch_user_api.h"
}


node::node()
{
    checkVersionGetNumSwitches(&mDeviceInfo);
}

node::~node()
{
    closeSwitchDevices();
}

//check version and return number of LWSwitches found 
int 
node::checkVersionGetNumSwitches(LWSWITCH_GET_DEVICES_V2_PARAMS *deviceInfo)
{
    LW_STATUS status;
    status = lwswitch_api_get_devices(deviceInfo);
    if (status != LW_OK)
    {
        if (status == LW_ERR_LIB_RM_VERSION_MISMATCH)
        {
            fprintf(stderr, "lwswitch-audit version is incompatible with LWSwitch driver. Please update with matching LWPU driver package");
            exit(-1);
        }
        // all other errors, log the error code and bail out
        fprintf(stderr, "lwswitch-audit:failed to query device information from LWSwitch driver, return status:%s\n", lwstatusToString(status));
        exit(-1);
    }
    if (deviceInfo->deviceCount <= 0)
    {
        fprintf(stderr, "No LWSwitches found\n");
        exit(-1);
    }

    // only count devices that are not degraded or excluded
    int deviceCount = 0;
    for (uint32_t i = 0; i < deviceInfo->deviceCount; i++)
    {
        if (deviceInfo->info[i].deviceReason == LWSWITCH_DEVICE_BLACKLIST_REASON_NONE)
        {
            deviceCount++;
        }
    }

    return deviceCount;
}

bool
node::readTables(naNodeTables_t &tables, int numSwitches, uint32_t maxTableEntries, bool isReqTable)
{
    int switchId;
    int validOutOfRangeEntry;
    for(int devId = 0; devId < numSwitches ; devId++) 
    {
        lwswitch* switchPtr = getSwitchAtDevId(devId);
        if (switchPtr == NULL) {
            // the switch is not opened, the switch could be missing or excluded
            continue;
        }

        switchId = switchPhyIDtoSwitchID(switchPtr->getPhyId());
        uint64_t portMask;
        if(switchPtr->readPortMask(portMask) == false)
        {
            return false;
        }
        
        //for all access ports
        for(int i = 0; i < getNumAccessPortsPerSwitch(); i++)
        {
            uint32_t switchPort = getNthAccessPort(switchId, i);
            PRINT_VERBOSE("\nSwitchId %d Switch Port %d ", switchId, switchPort);
            validOutOfRangeEntry=0;
            if(switchPtr->isPortEnabled(portMask, switchPort) )
                if ( ((isReqTable == true) && 
                      !switchPtr->readRequestTable(switchPort, tables[devId][switchPort], maxTableEntries, validOutOfRangeEntry, this)) ||
                     ((isReqTable == false) && 
                      !switchPtr->readResponseTable(switchPort, tables[devId][switchPort], maxTableEntries, validOutOfRangeEntry, this))
                   )
                {
                    fprintf(stderr, "Unable to read Table for LWSwitch=%d Port=%d\n", switchId, switchPort);
                    return false;
                }
            if(validOutOfRangeEntry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Request entries found for GPU IDS greater than %d\n", 
                        switchId, switchPort, validOutOfRangeEntry, getMaxGpu());
            }
        }

        //for all trunk ports
        for(int i = 0; i < getNumTrunkPortsPerSwitch(); i++)
        {
            uint32_t switchPort;
            switchPort = getNthTrunkPort(switchId, i);
            validOutOfRangeEntry=0;

            PRINT_VERBOSE("\nSwitchId %d Switch Port %d ", switchId, switchPort);
            if(switchPtr->isPortEnabled(portMask, switchPort) )
                if ( ((isReqTable == true) &&
                      !switchPtr->readRequestTable(switchPort, tables[devId][switchPort], maxTableEntries, validOutOfRangeEntry, this)) ||
                     ((isReqTable == false) &&
                      !switchPtr->readResponseTable(switchPort, tables[devId][switchPort], maxTableEntries, validOutOfRangeEntry, this))
                   )
                {
                    fprintf(stderr, "Unable to read Table for LWSwitch=%d Port=%d\n", switchId, switchPort);
                    return false;
                }
            if(validOutOfRangeEntry)
            {
                fprintf(stderr, "\t[Error] LWSwitch=%d port=%d, %d valid Request entries found for GPU IDS greater than %d\n", 
                        switchId, switchPort, validOutOfRangeEntry, getMaxGpu());
            }
        }
    }
    return true;
}
/*
devId: Switch instance in /dev/
Return: Corresponding switch id in HWLinks table
*/
int 
node::getDevToSwitchID(int devId)
{
    std::map<int, int>::iterator it;
    if ( (it = devIdToSwitchId.find(devId)) != devIdToSwitchId.end() ) 
    {
        return it->second;
    } 
    else
    {
        return -1;
    }
}

/*
switchId: Switch ID in HWLinks table
Return: Corresponding switch instance in /dev/
*/
int
node::getSwitchToDevID(int switchId)
{
    std::map<int, int>::iterator it;
    if ( (it = swIdToDevId.find(switchId)) != swIdToDevId.end() )
    {
        return it->second;
    }
    else
    {
        return -1;
    }
}

lwswitch*
node::getSwitchAtDevId(int devId)
{
    lwswitch* switchPtr = NULL;

    std::map<int, lwswitch*>::iterator it = allSwitches.find(devId);
    if (it != allSwitches.end()) {
        switchPtr = it->second;
    }

    return switchPtr;
}

int
node::readReqLinkId(int switchId, uint32_t switchPort)
{
    int devId = getSwitchToDevID(switchId);
    lwswitch* switchPtr = getSwitchAtDevId(devId);

    if (switchPtr) {
        return switchPtr->readReqLinkId(switchPort);
    } else {
        return -1;
    }
}

bool
node::openSwitchDevices()
{
    PRINT_VERBOSE("devId\tphyid\tswitchId\n");
    for(int devId = 0; devId < (int)mDeviceInfo.deviceCount; devId++)
    {
        LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo = mDeviceInfo.info[ devId ];

        if (switchInfo.deviceReason != LWSWITCH_DEVICE_BLACKLIST_REASON_NONE)
        {
            // the lwswitch is degraded or excluded
            // open and ioctls would fail on degraded or excluded lwswitch
            continue;
        }

        lwswitch *switchPtr = allocSwitch(switchInfo);
        allSwitches.insert(std::make_pair(devId, switchPtr));

        int phyId = switchPtr->getPhyId();
        int switchId = switchPhyIDtoSwitchID(phyId);
        PRINT_VERBOSE("%d\t%d\t%d\n", devId, phyId, switchId);
        setDevToSwitchMap(devId, switchId);
        setSwitchToDevMap(switchId, devId);
    }
    PRINT_VERBOSE("num switches found = %d\n", (int)allSwitches.size());
    return true;

}

void 
node::closeSwitchDevices()
{
    //walk vector and close all switches
    for (auto it = allSwitches.begin(); it != allSwitches.end(); ++it)
    {
        lwswitch *switchPtr = it->second;
        delete switchPtr;
    }
}
#ifdef DEBUG
bool
node::setRequestEntry(int switchId, int switchPort, int destGpuId, int valid, int egressPort)
{
    int devId = getSwitchToDevID(switchId);
    lwswitch* switchPtr = getSwitchAtDevId(devId);

    if (switchPtr) {
        return switchPtr->setRequestEntry(switchPort, destGpuId, valid, egressPort);
    } else {
        return false;
    }
}

bool
node::setResponseEntry(int switchId, int switchPort, int destRlid, int valid, int egressPort)
{
    int devId = getSwitchToDevID(switchId);
    lwswitch* switchPtr = getSwitchAtDevId(devId);

    if (switchPtr) {
        return switchPtr->setRequestEntry(switchPort, destRlid, valid, egressPort);
    } else {
        return false;
    }
}

#endif
