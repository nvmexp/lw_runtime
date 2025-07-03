#pragma once
#include "lwswitch_audit_node.h"
#include "lwswitch_audit_lwswitch.h"
#include "lwswitch_audit_willow.h"
#include <stdint.h>

extern "C"
{
#include "lwswitch_user_api.h"
}

class explorer16Juno : public node{
    public:
    static const int numAccessPorts = 96;
    private:
    static const int numGpu = 16;
    static const int numSwitch = 12;
    static const int numSwitchPerBaseboard = 6;
    static const int numTrunkPortsPerSwitch = 8;
    static const int numAccessPortsPerSwitch = 8;
    static const int numSwitchPorts = 36;

    uint32_t   trunkPortsNear[numTrunkPortsPerSwitch] = {
                                        0,
                                        1,
                                        2,
                                        3,
                                        8,
                                        9,
                                        10,
                                        11
                                    };
    uint32_t   trunkPortsFar[numTrunkPortsPerSwitch] =  {
                                        3,
                                        2,
                                        1,
                                        0,
                                        11,
                                        10,
                                        9,
                                        8
                                    };
    HWLink16_t HWLinks16[numAccessPorts] = {
                    {GB1_SW1_G1,   4,   GB1__GPU4,     1},
                    {GB1_SW1_G1,   5,   GB1__GPU1,     1},
                    {GB1_SW1_G1,   12,  GB1__GPU8,     1},
                    {GB1_SW1_G1,   13,  GB1__GPU5,     1},
                    {GB1_SW1_G1,   14,  GB1__GPU6,     1},
                    {GB1_SW1_G1,   15,  GB1__GPU7,     1},
                    {GB1_SW1_G1,   16,  GB1__GPU2,     1},
                    {GB1_SW1_G1,   17,  GB1__GPU3,     1},
                    {GB1_SW2_G2,   4,   GB1__GPU3,     5},
                    {GB1_SW2_G2,   5,   GB1__GPU8,     5},
                    {GB1_SW2_G2,   6,   GB1__GPU1,     5},
                    {GB1_SW2_G2,   7,   GB1__GPU4,     4},
                    {GB1_SW2_G2,   12,  GB1__GPU6,     4},
                    {GB1_SW2_G2,   13,  GB1__GPU7,     4},
                    {GB1_SW2_G2,   16,  GB1__GPU5,     5},
                    {GB1_SW2_G2,   17,  GB1__GPU2,     4},
                    {GB1_SW3_G3,   4,   GB1__GPU6,     3},
                    {GB1_SW3_G3,   5,   GB1__GPU7,     3},
                    {GB1_SW3_G3,   6,   GB1__GPU2,     2},
                    {GB1_SW3_G3,   7,   GB1__GPU4,     2},
                    {GB1_SW3_G3,   13,  GB1__GPU5,     3},
                    {GB1_SW3_G3,   14,  GB1__GPU8,     4},
                    {GB1_SW3_G3,   15,  GB1__GPU1,     3},
                    {GB1_SW3_G3,   17,  GB1__GPU3,     2},
                    {GB1_SW4_G4,   4,   GB1__GPU8,     0},
                    {GB1_SW4_G4,   5,   GB1__GPU6,     0},
                    {GB1_SW4_G4,   12,  GB1__GPU7,     0},
                    {GB1_SW4_G4,   13,  GB1__GPU5,     0},
                    {GB1_SW4_G4,   14,  GB1__GPU3,     0},
                    {GB1_SW4_G4,   15,  GB1__GPU2,     0},
                    {GB1_SW4_G4,   16,  GB1__GPU1,     0},
                    {GB1_SW4_G4,   17,  GB1__GPU4,     0},
                    {GB1_SW5_G5,   5,   GB1__GPU7,     2},
                    {GB1_SW5_G5,   6,   GB1__GPU3,     4},
                    {GB1_SW5_G5,   7,   GB1__GPU2,     5},
                    {GB1_SW5_G5,   12,  GB1__GPU8,     3},
                    {GB1_SW5_G5,   13,  GB1__GPU5,     2},
                    {GB1_SW5_G5,   14,  GB1__GPU6,     5},
                    {GB1_SW5_G5,   15,  GB1__GPU1,     4},
                    {GB1_SW5_G5,   17,  GB1__GPU4,     5},
                    {GB1_SW6_G6,   5,   GB1__GPU6,     2},
                    {GB1_SW6_G6,   6,   GB1__GPU1,     2},
                    {GB1_SW6_G6,   7,   GB1__GPU4,     3},
                    {GB1_SW6_G6,   12,  GB1__GPU2,     3},
                    {GB1_SW6_G6,   13,  GB1__GPU3,     3},
                    {GB1_SW6_G6,   14,  GB1__GPU5,     4},
                    {GB1_SW6_G6,   15,  GB1__GPU8,     2},
                    {GB1_SW6_G6,   17,  GB1__GPU7,     5},
                    {GB2_SW1_G1,   4,   GB2__GPU4,     1},
                    {GB2_SW1_G1,   5,   GB2__GPU1,     1},
                    {GB2_SW1_G1,   12,  GB2__GPU8,     1},
                    {GB2_SW1_G1,   13,  GB2__GPU5,     1},
                    {GB2_SW1_G1,   14,  GB2__GPU6,     1},
                    {GB2_SW1_G1,   15,  GB2__GPU7,     1},
                    {GB2_SW1_G1,   16,  GB2__GPU2,     1},
                    {GB2_SW1_G1,   17,  GB2__GPU3,     1},
                    {GB2_SW2_G2,   4,   GB2__GPU3,     5},
                    {GB2_SW2_G2,   5,   GB2__GPU8,     5},
                    {GB2_SW2_G2,   6,   GB2__GPU1,     5},
                    {GB2_SW2_G2,   7,   GB2__GPU4,     4},
                    {GB2_SW2_G2,   12,  GB2__GPU6,     4},
                    {GB2_SW2_G2,   13,  GB2__GPU7,     4},
                    {GB2_SW2_G2,   16,  GB2__GPU5,     5},
                    {GB2_SW2_G2,   17,  GB2__GPU2,     4},
                    {GB2_SW3_G3,   4,   GB2__GPU6,     3},
                    {GB2_SW3_G3,   5,   GB2__GPU7,     3},
                    {GB2_SW3_G3,   6,   GB2__GPU2,     2},
                    {GB2_SW3_G3,   7,   GB2__GPU4,     2},
                    {GB2_SW3_G3,   13,  GB2__GPU5,     3},
                    {GB2_SW3_G3,   14,  GB2__GPU8,     4},
                    {GB2_SW3_G3,   15,  GB2__GPU1,     3},
                    {GB2_SW3_G3,   17,  GB2__GPU3,     2},
                    {GB2_SW4_G4,   4,   GB2__GPU8,     0},
                    {GB2_SW4_G4,   5,   GB2__GPU6,     0},
                    {GB2_SW4_G4,   12,  GB2__GPU7,     0},
                    {GB2_SW4_G4,   13,  GB2__GPU5,     0},
                    {GB2_SW4_G4,   14,  GB2__GPU3,     0},
                    {GB2_SW4_G4,   15,  GB2__GPU2,     0},
                    {GB2_SW4_G4,   16,  GB2__GPU1,     0},
                    {GB2_SW4_G4,   17,  GB2__GPU4,     0},
                    {GB2_SW5_G5,   5,   GB2__GPU7,     2},
                    {GB2_SW5_G5,   6,   GB2__GPU3,     4},
                    {GB2_SW5_G5,   7,   GB2__GPU2,     5},
                    {GB2_SW5_G5,   12,  GB2__GPU8,     3},
                    {GB2_SW5_G5,   13,  GB2__GPU5,     2},
                    {GB2_SW5_G5,   14,  GB2__GPU6,     5},
                    {GB2_SW5_G5,   15,  GB2__GPU1,     4},
                    {GB2_SW5_G5,   17,  GB2__GPU4,     5},
                    {GB2_SW6_G6,   5,   GB2__GPU6,     2},
                    {GB2_SW6_G6,   6,   GB2__GPU1,     2},
                    {GB2_SW6_G6,   7,   GB2__GPU4,     3},
                    {GB2_SW6_G6,   12,  GB2__GPU2,     3},
                    {GB2_SW6_G6,   13,  GB2__GPU3,     3},
                    {GB2_SW6_G6,   14,  GB2__GPU5,     4},
                    {GB2_SW6_G6,   15,  GB2__GPU8,     2},
                    {GB2_SW6_G6,   17,  GB2__GPU7,     5},
                         };

    public:
    explorer16Juno();
    ~explorer16Juno();

    virtual uint32_t getMaxGpu() { return numGpu; }
    virtual int getMaxSwitch() { return numSwitch; }
    virtual int getMaxSwitchPerBaseboard() { return numSwitchPerBaseboard; }
    virtual int getReqEntriesPerGpu() { return willow::getReqEntriesPerGpu(); }
    virtual int getNumTrunkPortsPerSwitch() { return numTrunkPortsPerSwitch; }
    virtual int getNumAccessPortsPerSwitch() { return numAccessPortsPerSwitch; }
    virtual int getNumSwitchPorts() { return willow::getNumSwitchPorts(); };
    virtual int getNumReqIds() { return numGpu; }
    virtual int getNumResIds() { return numGpu * numSwitch; }

    virtual uint32_t getNthAccessPort(int switchId, unsigned int n);
    virtual uint32_t getNthTrunkPort(int switchId, unsigned int n);
    virtual int getConnectedGpuId(int switchId, uint32_t switchPort);
    virtual uint32_t getConnectedTrunkPortId(int switchId, uint32_t switchPort);
    virtual int switchPhyIDtoSwitchID(int phyId);
    virtual int isTrunkPort(unsigned int switchPort);
    virtual int getSrcPortId(int switchId, uint32_t switchPort, int srcGpuId);
    virtual lwswitch *allocSwitch( LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo );

};
