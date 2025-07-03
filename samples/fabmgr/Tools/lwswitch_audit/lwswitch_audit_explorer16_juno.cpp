#include "lwswitch_audit_explorer16_juno.h"
#include "lwswitch_audit_willow.h"

extern "C"
{
#include "lwswitch_user_api.h"
}

explorer16Juno::explorer16Juno()
{
}
explorer16Juno::~explorer16Juno()
{
}
uint32_t 
explorer16Juno::getNthAccessPort(int switchId, unsigned int n)
{
    return HWLinks16[switchId * getNumAccessPortsPerSwitch() + n].switchPort;
}

uint32_t
explorer16Juno::getNthTrunkPort(int switchId, unsigned int n)
{
    if (switchId < getMaxSwitchPerBaseboard())
        return trunkPortsNear[n];
    else
        return trunkPortsFar[n];
}

int
explorer16Juno::getConnectedGpuId(int switchId, uint32_t switchPort)
{
    HWLink16 *entry=NULL;
    int i;
    entry = &HWLinks16[switchId * getNumAccessPortsPerSwitch()];
    for (i = 0; i < getNumAccessPortsPerSwitch(); i++)
    {
        if (entry[i].switchPort == switchPort)
            return entry[i].GPUIndex;
    }
    return getMaxGpu() ;
}

uint32_t
explorer16Juno::getConnectedTrunkPortId(int switchId, uint32_t switchPort)
{
    uint32_t *localTrunkPort, *connectedTrunkPorts;
    if(switchId < getMaxSwitchPerBaseboard())
    {
        localTrunkPort = trunkPortsNear;
        connectedTrunkPorts = trunkPortsFar;
    }
    else
    {
        localTrunkPort = trunkPortsFar;
        connectedTrunkPorts = trunkPortsNear;
    }

    for(int i = 0; i < getNumTrunkPortsPerSwitch(); i++)
    {
        if(localTrunkPort[i] == switchPort)
            return connectedTrunkPorts[i];
    }
    return numSwitchPorts;
}

int
explorer16Juno::switchPhyIDtoSwitchID(int phyId)
{
    if(phyId >= 0x08 && phyId <= 0x0d)
        return phyId - 0x08;
    else if (phyId >= 0x18 && phyId <= 0x1d)
        return (phyId - 0x18) + getMaxSwitchPerBaseboard();
    else
        return -1;
}

int
explorer16Juno::isTrunkPort(unsigned int switchPort)
{
    int i;
    for( i = 0; i < getNumTrunkPortsPerSwitch(); i++)
    {
        if (switchPort == trunkPortsNear[i])
        {
            return true;
        }
    }
    return false;
}

//Returns the an identifier for the src port of a request. Responses are routed
//back to this id, which for willow switches is the requestor link id for the src port
int
explorer16Juno::getSrcPortId(int switchId, uint32_t switchPort, int srcGpuId)
{
    return readReqLinkId(switchId, switchPort);
}

lwswitch *
explorer16Juno::allocSwitch( LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo )
{
    return new willow( switchInfo );
}

