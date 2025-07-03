#pragma once
#include <vector>
#include "lwswitch_audit_node.h"
extern "C"
{
#include "lwlink_lib_ctrl.h"
#include "lwswitch_user_api.h"
}


class lwswitch {
    int readPhyId();
    static void acquireFabricManagementCapability(lwswitch_device *pLWSwitchDev);

protected:
    int mDevId;
    int mPhyId;
    lwlink_pci_dev_info mPciInfo;
    lwswitch_device *mpLWSwitchDev;
    uint32_t mArchType;

public:
    static lwswitch_device * openSwitchDev(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo);
    static unsigned int getSwitchArchInfo(lwswitch_device *pLWSwitchDev);
    lwswitch(LWSWITCH_DEVICE_INSTANCE_INFO_V2 &switchInfo);
    virtual ~lwswitch();
    int getPhyId() { return mPhyId;};
    int getDevId() { return mDevId;};

    virtual bool readRequestTable(uint32_t switchPort, naPortTable_t &reqTable,
                                  uint32_t maxTableEntries, int &validOutOfRangeEntry, node *np) = 0;
    virtual bool readResponseTable(uint32_t switchPort, naPortTable_t &resTable,
                                   uint32_t maxTableEntries, int &validOutOfRangeEntry, node *np) = 0;
#ifdef DEBUG
    virtual bool setRequestEntry(int switchPort, int destGpuId, int valid, int egressPort) = 0;
    virtual bool setResponseEntry(int switchPort, int destRlid, int valid, int egressPort) = 0;
#endif


    bool readPortMask(uint64_t &portMask);
    bool isPortEnabled(uint64_t mask, int portNum);
    int readReqLinkId( uint32_t switchPort);
};

