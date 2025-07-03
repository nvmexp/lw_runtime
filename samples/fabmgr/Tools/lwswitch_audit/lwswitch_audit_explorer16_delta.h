#pragma once
#include "lwswitch_audit_node.h"
#include "lwswitch_audit_lwswitch.h"
#include "lwswitch_audit_limerock.h"
#include <stdint.h>

extern "C"
{
#include "lwswitch_user_api.h"
}


class explorer16Delta : public node {
    private:
    static const int numAccessPorts = 192;
    static const int numAccessPortsPerSwitch = 16;
    static const int numTrunkPortsPerSwitch = 16;
    static const int numGpu = 16;
    static const int numSwitch = 12;
    static const int numSwitchPerBaseboard = 6;
    static const int numSwitchPorts = 36;

    uint32_t   trunkPortsNear[numTrunkPortsPerSwitch] = {
                                        0,
                                        1,
                                        2,
                                        3,
                                        4,
                                        5,
                                        6,
                                        7,
                                        16,
                                        17,
                                        18,
                                        19,
                                        20,
                                        21,
                                        22,
                                        23
                                    };
    uint32_t   trunkPortsFar[numTrunkPortsPerSwitch] =  {
                                        7,
                                        6,
                                        5,
                                        4,
                                        3,
                                        2,
                                        1,
                                        0,
                                        23,
                                        22,
                                        21,
                                        20,
                                        19,
                                        18,
                                        17,
                                        16
                                    };

    HWLink16 HWLinks16[numAccessPorts] = {
    { GB1_SW1_G1,    10,    GB1__GPU5,    2 },
    { GB1_SW1_G1,    11,    GB1__GPU5,    3 },
    { GB1_SW1_G1,    24,    GB1__GPU1,    3 },
    { GB1_SW1_G1,    25,    GB1__GPU1,    2 },
    { GB1_SW1_G1,    26,    GB1__GPU2,    3 },
    { GB1_SW1_G1,    27,    GB1__GPU2,    2 },
    { GB1_SW1_G1,    28,    GB1__GPU6,    3 },
    { GB1_SW1_G1,    29,    GB1__GPU6,    2 },
    { GB1_SW1_G1,    30,    GB1__GPU7,    2 },
    { GB1_SW1_G1,    31,    GB1__GPU7,    3 },
    { GB1_SW1_G1,    32,    GB1__GPU4,    2 },
    { GB1_SW1_G1,    33,    GB1__GPU4,    3 },
    { GB1_SW1_G1,    34,    GB1__GPU3,    2 },
    { GB1_SW1_G1,    35,    GB1__GPU3,    3 },
    { GB1_SW1_G1,    8,     GB1__GPU8,    2 },
    { GB1_SW1_G1,    9,     GB1__GPU8,    3 },
    { GB1_SW2_G2,    10,    GB1__GPU5,    9 },
    { GB1_SW2_G2,    11,    GB1__GPU5,    8 },
    { GB1_SW2_G2,    12,    GB1__GPU1,    8 },
    { GB1_SW2_G2,    13,    GB1__GPU1,    9 },
    { GB1_SW2_G2,    14,    GB1__GPU4,    9 },
    { GB1_SW2_G2,    15,    GB1__GPU4,    8 },
    { GB1_SW2_G2,    24,    GB1__GPU6,    8 },
    { GB1_SW2_G2,    25,    GB1__GPU6,    9 },
    { GB1_SW2_G2,    26,    GB1__GPU7,    9 },
    { GB1_SW2_G2,    27,    GB1__GPU7,    8 },
    { GB1_SW2_G2,    32,    GB1__GPU8,    8 },
    { GB1_SW2_G2,    33,    GB1__GPU8,    9 },
    { GB1_SW2_G2,    34,    GB1__GPU2,    8 },
    { GB1_SW2_G2,    35,    GB1__GPU2,    9 },
    { GB1_SW2_G2,    8,     GB1__GPU3,    8 },
    { GB1_SW2_G2,    9,     GB1__GPU3,    9 },
    { GB1_SW3_G3,    10,    GB1__GPU8,    5 },
    { GB1_SW3_G3,    11,    GB1__GPU8,    4 },
    { GB1_SW3_G3,    12,    GB1__GPU2,    5 },
    { GB1_SW3_G3,    13,    GB1__GPU2,    4 },
    { GB1_SW3_G3,    14,    GB1__GPU4,    4 },
    { GB1_SW3_G3,    15,    GB1__GPU4,    5 },
    { GB1_SW3_G3,    26,    GB1__GPU7,    4 },
    { GB1_SW3_G3,    27,    GB1__GPU7,    5 },
    { GB1_SW3_G3,    28,    GB1__GPU5,    4 },
    { GB1_SW3_G3,    29,    GB1__GPU5,    5 },
    { GB1_SW3_G3,    30,    GB1__GPU1,    4 },
    { GB1_SW3_G3,    31,    GB1__GPU1,    5 },
    { GB1_SW3_G3,    34,    GB1__GPU3,    4 },
    { GB1_SW3_G3,    35,    GB1__GPU3,    5 },
    { GB1_SW3_G3,    8,     GB1__GPU6,    4 },
    { GB1_SW3_G3,    9,     GB1__GPU6,    5 },
    { GB1_SW4_G4,    10,    GB1__GPU6,    0 },
    { GB1_SW4_G4,    11,    GB1__GPU6,    1 },
    { GB1_SW4_G4,    24,    GB1__GPU7,    1 },
    { GB1_SW4_G4,    25,    GB1__GPU7,    0 },
    { GB1_SW4_G4,    26,    GB1__GPU5,    1 },
    { GB1_SW4_G4,    27,    GB1__GPU5,    0 },
    { GB1_SW4_G4,    28,    GB1__GPU3,    1 },
    { GB1_SW4_G4,    29,    GB1__GPU3,    0 },
    { GB1_SW4_G4,    30,    GB1__GPU2,    0 },
    { GB1_SW4_G4,    31,    GB1__GPU2,    1 },
    { GB1_SW4_G4,    32,    GB1__GPU8,    0 },
    { GB1_SW4_G4,    33,    GB1__GPU8,    1 },
    { GB1_SW4_G4,    34,    GB1__GPU4,    0 },
    { GB1_SW4_G4,    35,    GB1__GPU4,    1 },
    { GB1_SW4_G4,    8,     GB1__GPU1,    1 },
    { GB1_SW4_G4,    9,     GB1__GPU1,    0 },
    { GB1_SW5_G5,    10,    GB1__GPU7,    10 },
    { GB1_SW5_G5,    11,    GB1__GPU7,    11 },
    { GB1_SW5_G5,    12,    GB1__GPU3,    10 },
    { GB1_SW5_G5,    13,    GB1__GPU3,    11 },
    { GB1_SW5_G5,    14,    GB1__GPU2,    11 },
    { GB1_SW5_G5,    15,    GB1__GPU2,    10 },
    { GB1_SW5_G5,    24,    GB1__GPU8,    10 },
    { GB1_SW5_G5,    25,    GB1__GPU8,    11 },
    { GB1_SW5_G5,    26,    GB1__GPU5,    11 },
    { GB1_SW5_G5,    27,    GB1__GPU5,    10 },
    { GB1_SW5_G5,    28,    GB1__GPU6,    11 },
    { GB1_SW5_G5,    29,    GB1__GPU6,    10 },
    { GB1_SW5_G5,    30,    GB1__GPU1,    11 },
    { GB1_SW5_G5,    31,    GB1__GPU1,    10 },
    { GB1_SW5_G5,    34,    GB1__GPU4,    10 },
    { GB1_SW5_G5,    35,    GB1__GPU4,    11 },
    { GB1_SW6_G6,    10,    GB1__GPU6,    6 },
    { GB1_SW6_G6,    11,    GB1__GPU6,    7 },
    { GB1_SW6_G6,    12,    GB1__GPU1,    6 },
    { GB1_SW6_G6,    13,    GB1__GPU1,    7 },
    { GB1_SW6_G6,    14,    GB1__GPU4,    7 },
    { GB1_SW6_G6,    15,    GB1__GPU4,    6 },
    { GB1_SW6_G6,    24,    GB1__GPU2,    7 },
    { GB1_SW6_G6,    25,    GB1__GPU2,    6 },
    { GB1_SW6_G6,    26,    GB1__GPU3,    6 },
    { GB1_SW6_G6,    27,    GB1__GPU3,    7 },
    { GB1_SW6_G6,    28,    GB1__GPU5,    7 },
    { GB1_SW6_G6,    29,    GB1__GPU5,    6 },
    { GB1_SW6_G6,    30,    GB1__GPU7,    7 },
    { GB1_SW6_G6,    31,    GB1__GPU7,    6 },
    { GB1_SW6_G6,    34,    GB1__GPU8,    6 },
    { GB1_SW6_G6,    35,    GB1__GPU8,    7 },
    { GB2_SW1_G1,    10,    GB2__GPU5,    2 },
    { GB2_SW1_G1,    11,    GB2__GPU5,    3 },
    { GB2_SW1_G1,    24,    GB2__GPU1,    3 },
    { GB2_SW1_G1,    25,    GB2__GPU1,    2 },
    { GB2_SW1_G1,    26,    GB2__GPU2,    3 },
    { GB2_SW1_G1,    27,    GB2__GPU2,    2 },
    { GB2_SW1_G1,    28,    GB2__GPU6,    3 },
    { GB2_SW1_G1,    29,    GB2__GPU6,    2 },
    { GB2_SW1_G1,    30,    GB2__GPU7,    2 },
    { GB2_SW1_G1,    31,    GB2__GPU7,    3 },
    { GB2_SW1_G1,    32,    GB2__GPU4,    2 },
    { GB2_SW1_G1,    33,    GB2__GPU4,    3 },
    { GB2_SW1_G1,    34,    GB2__GPU3,    2 },
    { GB2_SW1_G1,    35,    GB2__GPU3,    3 },
    { GB2_SW1_G1,    8,     GB2__GPU8,    2 },
    { GB2_SW1_G1,    9,     GB2__GPU8,    3 },
    { GB2_SW2_G2,    10,    GB2__GPU5,    9 },
    { GB2_SW2_G2,    11,    GB2__GPU5,    8 },
    { GB2_SW2_G2,    12,    GB2__GPU1,    8 },
    { GB2_SW2_G2,    13,    GB2__GPU1,    9 },
    { GB2_SW2_G2,    14,    GB2__GPU4,    9 },
    { GB2_SW2_G2,    15,    GB2__GPU4,    8 },
    { GB2_SW2_G2,    24,    GB2__GPU6,    8 },
    { GB2_SW2_G2,    25,    GB2__GPU6,    9 },
    { GB2_SW2_G2,    26,    GB2__GPU7,    9 },
    { GB2_SW2_G2,    27,    GB2__GPU7,    8 },
    { GB2_SW2_G2,    32,    GB2__GPU8,    8 },
    { GB2_SW2_G2,    33,    GB2__GPU8,    9 },
    { GB2_SW2_G2,    34,    GB2__GPU2,    8 },
    { GB2_SW2_G2,    35,    GB2__GPU2,    9 },
    { GB2_SW2_G2,    8,     GB2__GPU3,    8 },
    { GB2_SW2_G2,    9,     GB2__GPU3,    9 },
    { GB2_SW3_G3,    10,    GB2__GPU8,    5 },
    { GB2_SW3_G3,    11,    GB2__GPU8,    4 },
    { GB2_SW3_G3,    12,    GB2__GPU2,    5 },
    { GB2_SW3_G3,    13,    GB2__GPU2,    4 },
    { GB2_SW3_G3,    14,    GB2__GPU4,    4 },
    { GB2_SW3_G3,    15,    GB2__GPU4,    5 },
    { GB2_SW3_G3,    26,    GB2__GPU7,    4 },
    { GB2_SW3_G3,    27,    GB2__GPU7,    5 },
    { GB2_SW3_G3,    28,    GB2__GPU5,    4 },
    { GB2_SW3_G3,    29,    GB2__GPU5,    5 },
    { GB2_SW3_G3,    30,    GB2__GPU1,    4 },
    { GB2_SW3_G3,    31,    GB2__GPU1,    5 },
    { GB2_SW3_G3,    34,    GB2__GPU3,    4 },
    { GB2_SW3_G3,    35,    GB2__GPU3,    5 },
    { GB2_SW3_G3,    8,     GB2__GPU6,    4 },
    { GB2_SW3_G3,    9,     GB2__GPU6,    5 },
    { GB2_SW4_G4,    10,    GB2__GPU6,    0 },
    { GB2_SW4_G4,    11,    GB2__GPU6,    1 },
    { GB2_SW4_G4,    24,    GB2__GPU7,    1 },
    { GB2_SW4_G4,    25,    GB2__GPU7,    0 },
    { GB2_SW4_G4,    26,    GB2__GPU5,    1 },
    { GB2_SW4_G4,    27,    GB2__GPU5,    0 },
    { GB2_SW4_G4,    28,    GB2__GPU3,    1 },
    { GB2_SW4_G4,    29,    GB2__GPU3,    0 },
    { GB2_SW4_G4,    30,    GB2__GPU2,    0 },
    { GB2_SW4_G4,    31,    GB2__GPU2,    1 },
    { GB2_SW4_G4,    32,    GB2__GPU8,    0 },
    { GB2_SW4_G4,    33,    GB2__GPU8,    1 },
    { GB2_SW4_G4,    34,    GB2__GPU4,    0 },
    { GB2_SW4_G4,    35,    GB2__GPU4,    1 },
    { GB2_SW4_G4,    8,     GB2__GPU1,    1 },
    { GB2_SW4_G4,    9,     GB2__GPU1,    0 },
    { GB2_SW5_G5,    10,    GB2__GPU7,    10 },
    { GB2_SW5_G5,    11,    GB2__GPU7,    11 },
    { GB2_SW5_G5,    12,    GB2__GPU3,    10 },
    { GB2_SW5_G5,    13,    GB2__GPU3,    11 },
    { GB2_SW5_G5,    14,    GB2__GPU2,    11 },
    { GB2_SW5_G5,    15,    GB2__GPU2,    10 },
    { GB2_SW5_G5,    24,    GB2__GPU8,    10 },
    { GB2_SW5_G5,    25,    GB2__GPU8,    11 },
    { GB2_SW5_G5,    26,    GB2__GPU5,    11 },
    { GB2_SW5_G5,    27,    GB2__GPU5,    10 },
    { GB2_SW5_G5,    28,    GB2__GPU6,    11 },
    { GB2_SW5_G5,    29,    GB2__GPU6,    10 },
    { GB2_SW5_G5,    30,    GB2__GPU1,    11 },
    { GB2_SW5_G5,    31,    GB2__GPU1,    10 },
    { GB2_SW5_G5,    34,    GB2__GPU4,    10 },
    { GB2_SW5_G5,    35,    GB2__GPU4,    11 },
    { GB2_SW6_G6,    10,    GB2__GPU6,    6 },
    { GB2_SW6_G6,    11,    GB2__GPU6,    7 },
    { GB2_SW6_G6,    12,    GB2__GPU1,    6 },
    { GB2_SW6_G6,    13,    GB2__GPU1,    7 },
    { GB2_SW6_G6,    14,    GB2__GPU4,    7 },
    { GB2_SW6_G6,    15,    GB2__GPU4,    6 },
    { GB2_SW6_G6,    24,    GB2__GPU2,    7 },
    { GB2_SW6_G6,    25,    GB2__GPU2,    6 },
    { GB2_SW6_G6,    26,    GB2__GPU3,    6 },
    { GB2_SW6_G6,    27,    GB2__GPU3,    7 },
    { GB2_SW6_G6,    28,    GB2__GPU5,    7 },
    { GB2_SW6_G6,    29,    GB2__GPU5,    6 },
    { GB2_SW6_G6,    30,    GB2__GPU7,    7 },
    { GB2_SW6_G6,    31,    GB2__GPU7,    6 },
    { GB2_SW6_G6,    34,    GB2__GPU8,    6 },
    { GB2_SW6_G6,    35,    GB2__GPU8,    7 },
    };

    public:
    explorer16Delta();
    ~explorer16Delta();

    virtual uint32_t getMaxGpu() { return numGpu; }
    virtual int getMaxSwitch() { return numSwitch; }
    virtual int getMaxSwitchPerBaseboard() { return numSwitchPerBaseboard; }
    virtual int getReqEntriesPerGpu() { return limerock::getReqEntriesPerGpu();; }
    virtual int getNumTrunkPortsPerSwitch() { return numTrunkPortsPerSwitch; }
    virtual int getNumAccessPortsPerSwitch() { return numAccessPortsPerSwitch; }
    virtual int getNumSwitchPorts() { return limerock::getNumSwitchPorts(); };
    virtual uint32_t getNthAccessPort(int switchId, unsigned int n);
    virtual int getNumReqIds() { return numGpu; }
    virtual int getNumResIds() { return numGpu; }

    virtual uint32_t getNthTrunkPort(int switchId, unsigned int n);
    virtual int getConnectedGpuId(int switchId, uint32_t switchPort);
    virtual uint32_t getConnectedTrunkPortId(int switchId, uint32_t switchPort);
    virtual int switchPhyIDtoSwitchID(int phyId);
    virtual int isTrunkPort(unsigned int switchPort);
    virtual int getSrcPortId(int switchId, uint32_t switchPort, int srcGpuId);
    virtual lwswitch *allocSwitch( LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo );
};

