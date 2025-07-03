#pragma once
#include "lwswitch_audit_lwswitch.h"
#include <stdint.h>
#include <stdbool.h>

extern "C"
{
#include "lwswitch_user_api.h"
}

class limerock : public lwswitch {
    public:
    limerock(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo) : lwswitch(switchInfo) {};
    ~limerock(){};

    static int getNumSwitchPorts() { return 36; };
    virtual bool readRequestTable(uint32_t switchPort, naPortTable_t &reqTable,
                                  uint32_t maxTableEntries, int &validOutOfRangeEntry, node *np);
    virtual bool readResponseTable(uint32_t switchPort, naPortTable_t &resTable,
                                   uint32_t maxTableEntries, int &validOutOfRangeResEntry, node *np);
#ifdef DEBUG
    //TODO not supported on LR10
    virtual bool setRequestEntry(int switchPort, int destGpuId, int valid, int egressPort) ;
    virtual bool setResponseEntry(int switchPort, int destRlid, int valid, int egressPort) ;
#endif
    static int getReqEntriesPerGpu() { return 2; };
    private:
    uint32_t getRidTableSize() { return 512; }
    uint32_t getRemapTableSize() { return 2048; }

};

