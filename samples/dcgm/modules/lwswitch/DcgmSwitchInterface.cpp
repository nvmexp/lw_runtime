#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <sys/ioctl.h>
#include <stdexcept>

#include "DcgmGlobalFabricManager.h"
#include "DcgmLocalFabricManager.h"
#include "DcgmSwitchInterface.h"

#include "logging.h"

/*****************************************************************************/
/*  This is a thread that gets spawned per-switch by Local Fabric Manager.   */      
/*  It operates on a vector of IOCTL pointers, which it issues serially to   */
/*  its instance of the switch driver.                                       */
/*****************************************************************************/

DcgmSwitchInterface::DcgmSwitchInterface( uint32_t deviceInstance )
{
    mSwitchIndex = deviceInstance;

    mFileDescriptor = lwswitch_api_open_device(deviceInstance);
    if (mFileDescriptor < 0)
    {
        std::ostringstream ss;
        ss << "failed to open LWSwitch device index " << mSwitchIndex;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // the switch info to determine switch mArch
    if ( !fetchSwitchInfo() )
    {
        std::ostringstream ss;
        ss << "failed to get LWSwitch information for device index " << mSwitchIndex;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // update our gpio based switch physical id
    if ( !fetchSwitchPhysicalId() )
    {
        std::ostringstream ss;
        ss << "failed to get LWSwitch physical ID for device index " << mSwitchIndex;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    if ( !fetchSwitchPciInfo() )
    {
        std::ostringstream ss;
        ss << "failed to get LWSwitch PCI BDF information for device index " << mSwitchIndex;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    if ( !fetchSwitchPortInfo() )
    {
        std::ostringstream ss;
        ss << "failed to get LWSwitch port information for device index " << mSwitchIndex;
        PRINT_ERROR("%s", "%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }
};

DcgmSwitchInterface::~DcgmSwitchInterface()
{
    lwswitch_api_close_device(mFileDescriptor);
};

void DcgmSwitchInterface::dumpIoctlGetInfo( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INFO *ioctlParams = (LWSWITCH_GET_INFO *)ioctl->ioctlParams;
    uint32_t i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%5s %10s", "%5s %10s", "index", "info");

    for ( i = 0; i < ioctlParams->count; i++ )
    {
        PRINT_DEBUG("%5d 0x%08x %10d", "%5d 0x%08x %10d",
                    ioctlParams->index[i], ioctlParams->info[i], ioctlParams->info[i]);
    }
}

void DcgmSwitchInterface::dumpIoctlSwitchPortConfig( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_SWITCH_PORT_CONFIG *ioctlParams =
            (LWSWITCH_SET_SWITCH_PORT_CONFIG *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    PRINT_DEBUG("%d %d %d %d %d",
                "portNum %d, type %d, requesterLinkID %d, requesterLanID %d, count %d",
                ioctlParams->portNum, ioctlParams->type, ioctlParams->requesterLinkID,
                ioctlParams->requesterLanID, ioctlParams->count);
#else
    PRINT_DEBUG("%d %d %d",
                "portNum %d, requesterLinkID %d, type %d",
                ioctlParams->portNum, ioctlParams->requesterLinkID, ioctlParams->type);
#endif
}

void DcgmSwitchInterface::dumpIoctlGetIngressReqLinkID( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d", "portNum %d", ioctlParams->portNum);

    PRINT_DEBUG("%d", "requesterLinkID %d", ioctlParams->requesterLinkID);
}


void DcgmSwitchInterface::dumpIoctlGetIngressReq( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for (int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("%d, 0x%x, %d, 0x%x, 0x%x, 0x%x, %d",
                     "idx %d, mappedAddress 0x%x, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].idx,
                     ioctlParams->entries[i].entry.mappedAddress,
                     ioctlParams->entries[i].entry.routePolicy,
                     ioctlParams->entries[i].entry.vcModeValid7_0,
                     ioctlParams->entries[i].entry.vcModeValid15_8,
                     ioctlParams->entries[i].entry.vcModeValid17_16,
                     ioctlParams->entries[i].entry.entryValid);
    }
}


void DcgmSwitchInterface::dumpIoctlSetIngressReq( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_REQUEST_TABLE *ioctlParams =
            (LWSWITCH_SET_INGRESS_REQUEST_TABLE *)ioctl->ioctlParams;
    LWSWITCH_INGRESS_REQUEST_ENTRY  *entry;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("0x%x, %d, 0x%x, 0x%x, 0x%x, %d",
                     "mappedAddress 0x%x, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].mappedAddress,
                     ioctlParams->entries[i].routePolicy,
                     ioctlParams->entries[i].vcModeValid7_0,
                     ioctlParams->entries[i].vcModeValid15_8,
                     ioctlParams->entries[i].vcModeValid17_16,
                     ioctlParams->entries[i].entryValid);
    }
}

void DcgmSwitchInterface::dumpIoctlSetIngressValid( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_REQUEST_VALID *ioctlParams =
            (LWSWITCH_SET_INGRESS_REQUEST_VALID *)ioctl->ioctlParams;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("%d", "%d", ioctlParams->entryValid[i]);
    }
}

void DcgmSwitchInterface::dumpIoctlGetIngressResp( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for (int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {

        PRINT_DEBUG("%d, %d, 0x%x, 0x%x, 0x%x, %d",
                     "idx %d, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].idx,
                     ioctlParams->entries[i].entry.routePolicy,
                     ioctlParams->entries[i].entry.vcModeValid7_0,
                     ioctlParams->entries[i].entry.vcModeValid15_8,
                     ioctlParams->entries[i].entry.vcModeValid17_16,
                     ioctlParams->entries[i].entry.entryValid);
    }
}

void DcgmSwitchInterface::dumpIoctlSetIngressResp( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE *ioctlParams =
            (LWSWITCH_SET_INGRESS_RESPONSE_TABLE *)ioctl->ioctlParams;
    LWSWITCH_INGRESS_RESPONSE_ENTRY  *entry;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {

        PRINT_DEBUG("%d, 0x%x, 0x%x, 0x%x, %d",
                     "routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].routePolicy,
                     ioctlParams->entries[i].vcModeValid7_0,
                     ioctlParams->entries[i].vcModeValid15_8,
                     ioctlParams->entries[i].vcModeValid17_16,
                     ioctlParams->entries[i].entryValid);
    }
}

void DcgmSwitchInterface::dumpIoctlSetGangedLink( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_GANGED_LINK_TABLE *ioctlParams =
            (LWSWITCH_SET_GANGED_LINK_TABLE *)ioctl->ioctlParams;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("0x%x", "link_mask 0x%x", ioctlParams->link_mask);

    for ( i = 0; i < GANGED_LINK_TABLE_SIZE; i++ )
    {
        PRINT_DEBUG("%d 0x%x", "entries[%d] 0x%x", i, ioctlParams->entries[i]);
    }
}

void DcgmSwitchInterface::dumpIoctlGetInternalLatency( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INTERNAL_LATENCY *ioctlParams =
            (LWSWITCH_GET_INTERNAL_LATENCY *)ioctl->ioctlParams;
    uint32_t i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    for ( i = 0; i < mNumPorts; i++ )
    {
        if ( !(mEnabledPortMask & ((uint64_t)1 << i)) )
            continue;

        PRINT_DEBUG("%d", "port %d", i);
        if ( ioctlParams->egressHistogram[i].low != 0 )
            PRINT_DEBUG("%d %lld", "%d %lld", i, ioctlParams->egressHistogram[i].low);
        if ( ioctlParams->egressHistogram[i].medium != 0 )
            PRINT_DEBUG("%d %lld", "%d %lld", i, ioctlParams->egressHistogram[i].medium);
        if ( ioctlParams->egressHistogram[i].high != 0 )
            PRINT_DEBUG("%d %lld", "%d %lld", i, ioctlParams->egressHistogram[i].high);
        if ( ioctlParams->egressHistogram[i].panic != 0 )
            PRINT_DEBUG("%d %lld", "%d %lld", i, ioctlParams->egressHistogram[i].panic);
    }
}

void DcgmSwitchInterface::dumpIoctlSetLatencyBin( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_LATENCY_BINS *ioctlParams =
            (LWSWITCH_SET_LATENCY_BINS *)ioctl->ioctlParams;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%2s %12s %12s %11s", "%2s %12s %12s %11s",
                "VC", "lowThreshold", "medThreshold", "hiThreshold");

    for ( i = 0; i < LWSWITCH_MAX_VCS; i++ )
    {
        PRINT_DEBUG("%2d %12d %12d %11d", "%2d %12d %12d %11d",
                i,
                ioctlParams->bin[i].lowThreshold,
                ioctlParams->bin[i].medThreshold,
                ioctlParams->bin[i].hiThreshold);
    }
}

void DcgmSwitchInterface::dumpIoctlGetLwspliptCounters( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_LWLIPT_COUNTERS *ioctlParams =
            (LWSWITCH_GET_LWLIPT_COUNTERS *)ioctl->ioctlParams;
    uint32_t i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%4s %10s %10s %10s %10s", "%4s %10s %10s %10s %10s", "port", "txCounter0","rxCounter0", "txCounter1","rxCounter1");
    for ( i = 0; i < mNumPorts; i++ )
    {
        if ( !(mEnabledPortMask & ((uint64_t)1 << i)) )
            continue;
        
        PRINT_DEBUG("%4d %10llu %10llu %10llu %10llu", "%4d %10llu %10llu %10llu %10llu", i,
                    ioctlParams->liptCounter[i].txCounter0,
                    ioctlParams->liptCounter[i].rxCounter0,
                    ioctlParams->liptCounter[i].txCounter1,
                    ioctlParams->liptCounter[i].rxCounter1);
    }
}
void DcgmSwitchInterface::dumpIoctlGetLwspliptCounterConfig( switchIoctl_t *ioctl )
{
     LWSWITCH_GET_LWLIPT_COUNTER_CONFIG *ioctlParams =
            (LWSWITCH_GET_LWLIPT_COUNTER_CONFIG *)ioctl->ioctlParams;
    int i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }
}

void DcgmSwitchInterface::dumpIoctlSetLwspliptCounterConfig( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_LWLIPT_COUNTER_CONFIG *ioctlParams =
            (LWSWITCH_SET_LWLIPT_COUNTER_CONFIG *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }
}

void DcgmSwitchInterface::dumpIoctlGetErrors( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_ERRORS *ioctlParams = (LWSWITCH_GET_ERRORS *)ioctl->ioctlParams;
    uint32_t i;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%08x, %d", "errorMask 0x%08x, errorCount %d",
                ioctlParams->errorMask, ioctlParams->errorCount);

    PRINT_DEBUG("%4s %8s %3s %4s %7s %16s %8s", "%4s %8s %3s %4s %7s %16s %8s",
                "type", "severity", "src", "inst", "subinst", "time", "resolved");
    for ( i = 0; i < ioctlParams->errorCount; i++ )
    {
        PRINT_DEBUG("%4d %8d %3d %4d %7d %16llu %8d",
                    "%4d %8d %3d %4d %7d %16llu %8d",
                    ioctlParams->error[i].error_type, ioctlParams->error[i].severity,
                    ioctlParams->error[i].error_src, ioctlParams->error[i].instance,
                    ioctlParams->error[i].subinstance, ioctlParams->error[i].time,
                    ioctlParams->error[i].error_resolved);
    }
}

void DcgmSwitchInterface::dumpIoctlUnregisterLink( switchIoctl_t *ioctl )
{
    LWSWITCH_UNREGISTER_LINK_PARAMS *ioctlParams = (LWSWITCH_UNREGISTER_LINK_PARAMS *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d", "portNum %d", ioctlParams->portNum);
}

void DcgmSwitchInterface::dumpIoctlResetAndDrainLinks( switchIoctl_t *ioctl )
{
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS *ioctlParams = (LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("0x%llx", "linkMask  0x%llx", ioctlParams->linkMask);
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
DcgmSwitchInterface::dumpIoctlSetRemapPolicy( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_REMAP_POLICY *ioctlParams = (LWSWITCH_SET_REMAP_POLICY *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("%d, %d, %d, 0x%x, 0x%x, 0x%x, 0x%x",
                    "entryValid %d, targetId %d, irlSelect %d, flags 0x%x, reqCtxMask 0x%x, Chk 0x%x, Rep 0x%x",
                    ioctlParams->remapPolicy[i].entryValid,
                    ioctlParams->remapPolicy[i].targetId,
                    ioctlParams->remapPolicy[i].irlSelect,
                    ioctlParams->remapPolicy[i].flags,
                    ioctlParams->remapPolicy[i].reqCtxMask,
                    ioctlParams->remapPolicy[i].reqCtxChk,
                    ioctlParams->remapPolicy[i].reqCtxRep);

        PRINT_DEBUG("0x%llu, 0x%llu, 0x%llu, 0x%llu",
                    "address 0x%llu, Offset 0x%llu, Base 0x%llu, Limit 0x%llu",
                    ioctlParams->remapPolicy[i].address,
                    ioctlParams->remapPolicy[i].addressOffset,
                    ioctlParams->remapPolicy[i].addressBase,
                    ioctlParams->remapPolicy[i].addressLimit);
    }
}

void
DcgmSwitchInterface::dumpIoctlSetRoutingId( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_ROUTING_ID *ioctlParams = (LWSWITCH_SET_ROUTING_ID *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("%d: %d, %d, %d, %d",
                    "index %d: entryValid %d, useRoutingLan %d, enableIrlErrResponse %d, numEntries %d",
                    ioctlParams->firstIndex + i,
                    ioctlParams->routingId[i].entryValid,
                    ioctlParams->routingId[i].useRoutingLan,
                    ioctlParams->routingId[i].enableIrlErrResponse,
                    ioctlParams->routingId[i].numEntries);

        for ( int j = 0; j < (int)ioctlParams->routingId[i].numEntries; j++ )
        {
            PRINT_DEBUG("%d: %d, %d",
                        "index %d: vcMap %d, destPortNum %d",
                        j, ioctlParams->routingId[i].portList[j].vcMap,
                        ioctlParams->routingId[i].portList[j].destPortNum);
        }
    }
}

void
DcgmSwitchInterface::dumpIoctlSetRoutingLan( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_ROUTING_LAN *ioctlParams = (LWSWITCH_SET_ROUTING_LAN *)ioctl->ioctlParams;

    dumpIoctlDefault( ioctl );
    if ( !ioctlParams )
    {
        PRINT_DEBUG(" ", "Invalid params");
        return;
    }

    PRINT_DEBUG("%d %d %d",
                "portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        PRINT_DEBUG("%d: %d, %d",
                    "index %d: entryValid %d, numEntries %d",
                    ioctlParams->firstIndex + i,
                    ioctlParams->routingLan[i].entryValid,
                    ioctlParams->routingLan[i].numEntries);

        for ( int j = 0; j < (int)ioctlParams->routingLan[i].numEntries; j++ )
        {
            PRINT_DEBUG("%d: %d, %d",
                        "index %d: groupSelect %d, groupSize %d",
                        j, ioctlParams->routingLan[i].portList[j].groupSelect,
                        ioctlParams->routingLan[i].portList[j].groupSize);
        }
    }
}
#endif

void DcgmSwitchInterface::dumpIoctlDefault( switchIoctl_t *ioctl )
{
    PRINT_DEBUG("0x%x %p", "type 0x%x, params %p",
                ioctl->type, ioctl->ioctlParams);
}

void DcgmSwitchInterface::dumpIoctl( switchIoctl_t *ioctl )
{
    if ( !ioctl )
    {
        PRINT_DEBUG("", "Invalid ioctl");
        return;
    }

    switch ( ioctl->type )
    {
    case IOCTL_LWSWITCH_GET_INFO:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_INFO");
        dumpIoctlGetInfo( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG");
        dumpIoctlSwitchPortConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE");
        dumpIoctlSetIngressReq( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE");
        dumpIoctlGetIngressReq( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID");
        dumpIoctlSetIngressValid( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQLINKID:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_INGRESS_REQLINKID");
        dumpIoctlGetIngressReqLinkID( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE");
        dumpIoctlGetIngressResp( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE");
        dumpIoctlSetIngressResp( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE");
        dumpIoctlSetGangedLink( ioctl );
        break;

    // LwSwitch Performance Metric Control and Collection
    case IOCTL_LWSWITCH_GET_INTERNAL_LATENCY:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_INTERNAL_LATENCY");
        dumpIoctlGetInternalLatency( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_LATENCY_BINS:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_LATENCY_BINS");
        dumpIoctlSetLatencyBin( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTERS:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_LWLIPT_COUNTERS");
        dumpIoctlGetLwspliptCounters( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG");
        dumpIoctlGetLwspliptCounterConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG");
        dumpIoctlSetLwspliptCounterConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_ERRORS:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_GET_ERRORS");
        dumpIoctlGetErrors( ioctl );
        break;

    case IOCTL_LWSWITCH_UNREGISTER_LINK:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_UNREGISTER_LINK");
        dumpIoctlUnregisterLink( ioctl );
        break;

    case IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS:
         PRINT_DEBUG("", "IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS");
         dumpIoctlResetAndDrainLinks( ioctl );
         break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    case IOCTL_LWSWITCH_SET_REMAP_POLICY:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_REMAP_POLICY");
        dumpIoctlSetRemapPolicy( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_ROUTING_ID:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_ROUTING_ID");
        dumpIoctlSetRoutingId( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_ROUTING_LAN:
        PRINT_DEBUG("", "IOCTL_LWSWITCH_SET_ROUTING_LAN");
        dumpIoctlSetRoutingLan( ioctl );
        break;
#endif

    default:
        PRINT_DEBUG("0x%x", "Unknown ioctl->type 0x%x", ioctl->type);
        dumpIoctlDefault( ioctl );
        break;
    }
};

FM_ERROR_CODE
DcgmSwitchInterface::doIoctl( switchIoctl_t *pIoctl )
{
    int ret;

    switch ( pIoctl->type )
    {
    case IOCTL_LWSWITCH_GET_INFO:
        break;

    case IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG:
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE:
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE:
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID:
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQLINKID:
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE:
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE:
        break;

    case IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE:
        break;

    // LwSwitch Performance Metric Control and Collection
    case IOCTL_LWSWITCH_GET_INTERNAL_LATENCY:
        break;

    case IOCTL_LWSWITCH_SET_LATENCY_BINS:
        break;

    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTERS:
        break;

    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG:
        break;

    case IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG:
        break;

    // LwSwitch Error Data Control and Collection
    case IOCTL_LWSWITCH_GET_ERRORS:
        break;

    case IOCTL_LWSWITCH_UNREGISTER_LINK:
        break;

    case IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS:
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    // Limerock routing table
    case IOCTL_LWSWITCH_SET_REMAP_POLICY:
        break;

    case IOCTL_LWSWITCH_SET_ROUTING_ID:
        break;

    case IOCTL_LWSWITCH_SET_ROUTING_LAN:
        break;
#endif

    default:
        PRINT_ERROR("%d",
                    "ioctl type 0x%x not handled",
                    pIoctl->type);
        return FM_IOCTL_ERR;
    }

    ret = ioctl( mFileDescriptor, pIoctl->type, pIoctl->ioctlParams );

    if ( ret < 0 ) {
        PRINT_ERROR( "%d %d %d %d %d",
                     "LWSwitch driver ioctl failed: switchIdx %d physicalId %d type 0x%x return %d errno %d",
                     mSwitchIndex, mPhysicalId, pIoctl->type, ret, errno );
        return FM_IOCTL_ERR;
    }

    dumpIoctl( pIoctl );
    return FM_SUCCESS;
}

int
DcgmSwitchInterface::getFd()
{
    return mFileDescriptor;
}

uint32_t
DcgmSwitchInterface::getSwitchDevIndex()
{
    return mSwitchIndex;
}

uint32_t
DcgmSwitchInterface::getSwitchPhysicalId()
{
    return mPhysicalId;
}

const DcgmFMPciInfo&
DcgmSwitchInterface::getSwtichPciInfo()
{
    return mPciInfo;
}

uint32_t
DcgmSwitchInterface::getNumPorts()
{
    return mNumPorts;
}

uint64_t
DcgmSwitchInterface::getEnabledPortMask()
{
    return mEnabledPortMask;
}

/* update cached enabled port mask information after unregistering/disabling ports */
void
DcgmSwitchInterface::updateEnabledPortMask()
{
    fetchSwitchPortInfo();
}

bool
DcgmSwitchInterface::fetchSwitchPhysicalId()
{
    FM_ERROR_CODE rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_SUCCESS )
    {
        // failed to get the switch physical id information
        return false;
    } 

    // update our local switch physical id information
    mPhysicalId = ioctlParams.info[0];
    return true;
}

bool
DcgmSwitchInterface::fetchSwitchPciInfo()
{
    FM_ERROR_CODE rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 4;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PCI_DOMAIN;
    ioctlParams.index[1] = LWSWITCH_GET_INFO_INDEX_PCI_BUS;
    ioctlParams.index[2] = LWSWITCH_GET_INFO_INDEX_PCI_DEVICE;
    ioctlParams.index[3] = LWSWITCH_GET_INFO_INDEX_PCI_FUNCTION;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_SUCCESS )
    {
        // failed to get the switch PCI BDF information
        return false;
    } 

    // update our local PCI BDF information
    mPciInfo.domain = ioctlParams.info[0];
    mPciInfo.bus = ioctlParams.info[1];
    mPciInfo.device = ioctlParams.info[2];
    mPciInfo.function = ioctlParams.info[3];

    return true;
}

bool
DcgmSwitchInterface::fetchSwitchPortInfo()
{
    FM_ERROR_CODE rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 4;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_NUM_PORTS;
    ioctlParams.index[1] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0;
    ioctlParams.index[2] = LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32;
    ioctlParams.index[3] = LWSWITCH_GET_INFO_INDEX_NUM_VCS;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_SUCCESS )
    {
        // failed to get the switch port config items
        return false;
    } 

    // number of available ports 
    mNumPorts = ioctlParams.info[0];
    // enabled port mask is exposed as two 32bits.
    // generate 64bit mask by combining the two
    uint32_t portMaskLsb = ioctlParams.info[1];
    uint32_t portMaskMsb = ioctlParams.info[2];
    mEnabledPortMask = (uint64_t) portMaskMsb << 32 | portMaskLsb;
    // number of VCs
    mNumVCs =  ioctlParams.info[3];

    return true;
}

bool
DcgmSwitchInterface::fetchSwitchInfo()
{
    FM_ERROR_CODE rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;
    ioctlParams.count = 1;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_SUCCESS )
    {
        // failed to get the switch port config items
        return false;
    }

    // Get the switch arch
    mArch = ioctlParams.info[0];
    return true;
}

bool
DcgmSwitchInterface::isWillowSwitch()
{
    return (mArch == LWSWITCH_GET_INFO_INDEX_ARCH_SV10);
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
bool
DcgmSwitchInterface::isLimerockSwitch()
{
    return (mArch == LWSWITCH_GET_INFO_INDEX_ARCH_LR10);
}
#endif

