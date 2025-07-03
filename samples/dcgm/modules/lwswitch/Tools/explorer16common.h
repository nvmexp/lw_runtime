
#ifndef EXPLORER16_COMMON_H
#define EXPLORER16_COMMON_H
#include  <sys/types.h>
#include <stdint.h>

typedef enum switchID16_t {
    GB1_SW1_G1 = 0,
    GB1_SW2_G2 = 1,
    GB1_SW3_G3 = 2,
    GB1_SW4_G4 = 3,
    GB1_SW5_G5 = 4,
    GB1_SW6_G6 = 5,
    GB2_SW1_G1 = 6,
    GB2_SW2_G2 = 7,
    GB2_SW3_G3 = 8,
    GB2_SW4_G4 = 9,
    GB2_SW5_G5 = 10,
    GB2_SW6_G6 = 11
} switchID16;

typedef enum GPUID_16t {
    GB1__GPU1 = 0,
    GB1__GPU2 = 1,
    GB1__GPU3 = 2,
    GB1__GPU4 = 3,
    GB1__GPU5 = 4,
    GB1__GPU6 = 5,
    GB1__GPU7 = 6,
    GB1__GPU8 = 7,
    GB2__GPU1 = 8,
    GB2__GPU2 = 9,
    GB2__GPU3 = 10,
    GB2__GPU4 = 11,
    GB2__GPU5 = 12,
    GB2__GPU6 = 13,
    GB2__GPU7 = 14,
    GB2__GPU8 = 15,
} GPUID16;


typedef struct HWLink16_t {
    unsigned int willowIndex;
    unsigned int willowPort;
    int          GPUIndex;
    unsigned int GPUPort;
} HWLink16;

#define EXPLORER16_NUM_USED_PORTS       96

#define EXPLORER16_NUM_GPU              16
#define EXPLORER16_NUM_SWITCH           12
#define EXPLORER16_NUM_SWITCH_PER_BASEBOARD 6
#define EXPLORER16_REQ_ENTRIES_PER_GPU  4
#define EXPLORER16_NUM_TRUNK_PORTS      8
#define EXPLORER16_NUM_ACCESS_PORTS     8
#define EXPLORER16_NUM_SWITCH_PORTS     18
#define EXPLORER16_NUM_TABLE_ENTRIES    8192

extern HWLink16 HWLinks16[EXPLORER16_NUM_USED_PORTS];
extern uint32_t   trunkPortsNear[8];
extern uint32_t   trunkPortsFar[8];


// Note: The following shared LWSwitch VM partition information is created based on
// Application Note ID https://apps.lwinfo.lwpu.com/pid/contentlibraries/detail?id=1030696
// Table 5: Supported System Partitions: Shared LWSwitches.
struct SharedPartGpuInfoTable_t
{
    uint32_t physicalId;
    uint32_t numEnabledLinks;
    uint64_t enabledLinkMask;
};

struct SharedPartSwitchInfoTable_t
{
    uint32_t physicalId;
    uint32_t numEnabledLinks;
    uint64_t enabledLinkMask;
};

struct  SharedPartInfoTable_t {
    uint32_t partitionId;
    uint32_t gpuCount;
    uint32_t switchCount;
    uint32_t lwLinkIntraTrunkConnCount;
    uint32_t lwLinkInterTrunkConnCount;
    struct SharedPartGpuInfoTable_t     gpuInfo[EXPLORER16_NUM_GPU];
    struct SharedPartSwitchInfoTable_t  switchInfo[EXPLORER16_NUM_SWITCH];
};


#define MAX_SHARED_LWSWITCH_FABRIC_PARTITIONS 31

extern SharedPartInfoTable_t gSharedVMPartInfo[MAX_SHARED_LWSWITCH_FABRIC_PARTITIONS];

uint32_t exp16GetNthAccessPort(int switch_id, unsigned int n);
uint32_t exp16GetNthTrunkPort(int switch_id, unsigned int n);
int exp16GetConnectedGPUID(int switch_id, uint32_t willow_port);
int exp16ComputeReqLinkID(int switch_id, uint32_t willow_port);
uint32_t exp16GetConnectedTrunkPortId(int switch_id, uint32_t willow_port);
int exp16SwitchPhyIDtoSwitchID(int phy_id);
bool exp16IsTrunkPort(unsigned int willow_port);

#endif

