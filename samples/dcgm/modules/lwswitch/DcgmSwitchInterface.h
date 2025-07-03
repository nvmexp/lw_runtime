#ifndef DCGMSWITCHINTERFACE_H
#define DCGMSWITCHINTERFACE_H
 
#include <queue>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "ioctl_dev_lwswitch.h"
#include "DcgmFMError.h"

/*****************************************************************************/
/*  This is a thread that gets spawned per-switch by Local Fabric Manager.   */      
/*  It operates on a vector of IOCTL pointers, which it issues serially to   */
/*  its instance of the switch driver.                                       */
/*****************************************************************************/

typedef enum SwitchIoctlStatus_enum
{
    SWITCH_IOCTL_SUCCESS = 0,
    SWITCH_IOCTL_FAIL    = 1
} SwitchIoctlStatus_t;

typedef struct
{
    int    type;
    void  *ioctlParams;
} switchIoctl_struct;

typedef switchIoctl_struct switchIoctl_t;

typedef struct
{
    uint32_t          switchPhysicalId;
    LWSWITCH_ERROR    switchError;
} SwitchError_struct;

typedef struct
{
    uint32_t                       switchPhysicalId;
    LWSWITCH_GET_INTERNAL_LATENCY  latencies;
} SwitchLatency_struct;

typedef struct
{
    uint32_t                        switchPhysicalId;
    LWSWITCH_GET_LWLIPT_COUNTERS    counters;
} LwlinkCounter_struct;

typedef std::map<uint32_t, int> DcgmLWSwitchPhyIdToFdInfoMap;

/*****************************************************************************/

class DcgmSwitchInterface
{
public:
    DcgmSwitchInterface( uint32_t deviceInstance );
    ~DcgmSwitchInterface();

    FM_ERROR_CODE doIoctl( switchIoctl_t *ioctl );
    int getFd();
    uint32_t getSwitchDevIndex();
    uint32_t getSwitchPhysicalId();
    const DcgmFMPciInfo& getSwtichPciInfo();
    uint32_t getNumPorts();
    uint64_t getEnabledPortMask();
    void updateEnabledPortMask();
    bool isWillowSwitch();
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    bool isLimerockSwitch();
#endif

private:
    void dumpIoctl( switchIoctl_t *ioctl );
    void dumpIoctlDefault( switchIoctl_t *ioctl );

    void dumpIoctlGetInfo( switchIoctl_t *ioctl );
    void dumpIoctlSwitchConfig( switchIoctl_t *ioctl );
    void dumpIoctlSwitchPortConfig( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressReq( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressReq( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressValid( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressResp( switchIoctl_t *ioctl );
    void dumpIoctlSetIngressResp( switchIoctl_t *ioctl );
    void dumpIoctlSetGangedLink( switchIoctl_t *ioctl );
    void dumpIoctlGetIngressReqLinkID( switchIoctl_t *ioctl );

    void dumpIoctlGetInternalLatency( switchIoctl_t *ioctl );
    void dumpIoctlSetLatencyBin( switchIoctl_t *ioctl );
    void dumpIoctlGetLwspliptCounters( switchIoctl_t *ioctl );
    void dumpIoctlGetLwspliptCounterConfig( switchIoctl_t *ioctl );
    void dumpIoctlSetLwspliptCounterConfig( switchIoctl_t *ioctl );

    void dumpIoctlGetErrors( switchIoctl_t *ioctl );
    void dumpIoctlUnregisterLink( switchIoctl_t *ioctl );
    void dumpIoctlResetAndDrainLinks( switchIoctl_t *ioctl );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    void dumpIoctlSetRemapPolicy( switchIoctl_t *ioctl );
    void dumpIoctlSetRoutingId( switchIoctl_t *ioctl );
    void dumpIoctlSetRoutingLan( switchIoctl_t *ioctl );
#endif

    bool fetchSwitchPhysicalId();
    bool fetchSwitchPciInfo();
    bool fetchSwitchPortInfo();
    bool fetchSwitchInfo();

    uint32_t mSwitchIndex;
    int mFileDescriptor;
    uint32_t mPhysicalId;
    DcgmFMPciInfo mPciInfo;
    uint32_t mNumPorts;
    uint32_t mNumVCs;
    uint64_t mEnabledPortMask;
    uint32_t mArch;
};
#endif
