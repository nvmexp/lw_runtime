#pragma once
#include <stdint.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <map>
extern "C"
{
#include "lwswitch_user_api.h"
}


#define DEST_UNREACHABLE        INT_MAX
#define DEST_ERROR              -1


//per-port request/response table
typedef std::vector<int> naPortTable_t;

//all request/request tables for a switch
typedef std::vector<naPortTable_t> naSwitchTables_t;

//all request/response tables for all switches on a node
typedef std::vector<naSwitchTables_t> naNodeTables_t;

//paths matrix
typedef std::vector<std::vector<int>> naPathsMatrix_t;

typedef struct HWLink16 {
    unsigned int switchIndex;
    unsigned int switchPort;
    int          GPUIndex;
    unsigned int GPUPort;
} HWLink16_t;


class lwswitch;
class node {
    
    LWSWITCH_GET_DEVICES_V2_PARAMS mDeviceInfo;

    //map switch instance /dev/lwpu-lwswitch<dev id> to switchId (0-12 as used in HWLinks table)
    std::map<int, int> devIdToSwitchId;

    //map switchId (0-12 as used in HWLinks table) to switch instance /dev/lwpu-lwswitch<dev id>
    std::map<int, int> swIdToDevId;

    protected:
    std::map<int, lwswitch*> allSwitches;

    void setDevToSwitchMap(int devId, int switchId) { devIdToSwitchId[devId] = switchId;};

    void setSwitchToDevMap(int switchId, int devId) { swIdToDevId[switchId] = devId;}

    public:
    node(void);
    ~node(void);

    static int checkVersionGetNumSwitches(LWSWITCH_GET_DEVICES_V2_PARAMS *deviceInfo);

    virtual uint32_t getMaxGpu() = 0;
    virtual int getMaxSwitch() = 0;
    virtual int getMaxSwitchPerBaseboard() = 0;
    virtual int getReqEntriesPerGpu() = 0;
    virtual int getNumTrunkPortsPerSwitch() = 0;
    virtual int getNumAccessPortsPerSwitch() = 0;
    
    virtual int getNumSwitchPorts() = 0;
    virtual int getNumReqIds() = 0;
    virtual int getNumResIds() = 0;

    virtual bool openSwitchDevices();
    virtual void closeSwitchDevices();

    virtual uint32_t getNthAccessPort(int switchId, unsigned int n) = 0;
    virtual uint32_t getNthTrunkPort(int switchId, unsigned int n) = 0;
    virtual int getConnectedGpuId(int switchId, uint32_t switchPort) = 0;
    virtual uint32_t getConnectedTrunkPortId(int switchId, uint32_t switchPort) = 0;
    virtual int switchPhyIDtoSwitchID(int phyId) = 0;
    virtual int isTrunkPort(unsigned int switchPort) = 0;
    virtual int getSrcPortId(int switchId, uint32_t switchPort, int srcGpuId) = 0;
    virtual lwswitch *allocSwitch(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo) = 0;

    int readReqLinkId(int switchId, uint32_t switchPort);

    bool readTables(naNodeTables_t &tables, int numSwitches, uint32_t maxTableEntries, bool isReqTable);
    int getDevToSwitchID(int devId);
    int getSwitchToDevID(int switchId);
    lwswitch* getSwitchAtDevId(int devId);

#ifdef DEBUG
    bool setRequestEntry(int switchId, int switchPort, int destGpuId, int valid, int egressPort);
    bool setResponseEntry(int switchId, int switchPort, int destRlid, int valid, int egressPort);
#endif


};

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


