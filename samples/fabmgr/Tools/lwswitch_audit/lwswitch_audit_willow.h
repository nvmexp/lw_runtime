#pragma once
#include <stdint.h>
#include "lwswitch_audit_lwswitch.h"

extern "C"
{
#include "lwswitch_user_api.h"
}

class willow : public lwswitch {
    public:
    willow(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo):lwswitch(switchInfo) {};
    ~willow() {};

    static int getNumSwitchPorts() {return 18;}
    virtual bool readRequestTable(uint32_t switchPort, naPortTable_t &reqTable,
                                  uint32_t maxTableEntries, int &validOutOfRangeEntry, node *np);
    virtual bool readResponseTable(uint32_t switchPort, naPortTable_t &resTable,
                                   uint32_t maxTableEntries, int &validOutOfRangeResEntry, node *np);
#ifdef DEBUG
    virtual bool setRequestEntry(int switchPort, int destGpuId, int valid, int egressPort);
    virtual bool setResponseEntry(int switchPort, int destRlid, int valid, int egressPort);
#endif
    static int getReqEntriesPerGpu() { return 4; };
    private:
    uint32_t getReqTableSize() { return 8192; }
    uint32_t getResTableSize() { return 8192; }
};

