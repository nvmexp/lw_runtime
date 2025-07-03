#include "lwswitch_audit_explorer16_delta.h"
#include "lwswitch_audit_limerock.h"
#include "lwswitch_audit_logging.h"

extern "C"
{
#include "lwswitch_user_api.h"
}

explorer16Delta::explorer16Delta()
{
}

explorer16Delta::~explorer16Delta()
{
}

uint32_t 
explorer16Delta::getNthAccessPort(int switchId, unsigned int n)
{
    return HWLinks16[switchId * getNumAccessPortsPerSwitch() + n].switchPort;
}

uint32_t
explorer16Delta::getNthTrunkPort(int switchId, unsigned int n)
{
    if (switchId < getMaxSwitchPerBaseboard())
        return trunkPortsNear[n];
    else
        return trunkPortsFar[n];
}

int
explorer16Delta::getConnectedGpuId(int switchId, uint32_t switchPort)
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
explorer16Delta::getConnectedTrunkPortId(int switchId, uint32_t switchPort)
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
    return explorer16Delta::numSwitchPorts;
}

int
explorer16Delta::switchPhyIDtoSwitchID(int phyId)
{
    if(phyId >= 0x08 && phyId <= 0x0d)
        return phyId - 0x08;
    else if (phyId >= 0x18 && phyId <= 0x1d)
        return (phyId - 0x18) + getMaxSwitchPerBaseboard();
    else
        return -1;
}

int
explorer16Delta::isTrunkPort(unsigned int switchPort)
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
//back to this id, which for limerock switches is the targetId(srcGpuId) passed in
int
explorer16Delta::getSrcPortId(int switchId, uint32_t switchPort, int srcGpuId)
{
    return readReqLinkId(switchId, switchPort);
}

lwswitch *
explorer16Delta::allocSwitch( LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo )
{
    return new limerock( switchInfo );
}
