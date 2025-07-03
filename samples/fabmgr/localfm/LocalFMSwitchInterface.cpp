/*
 *  Copyright 2018-2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/sysmacros.h>
#include <sys/ioctl.h>
#include <stdexcept>

#include "GlobalFabricManager.h"
#include "LocalFabricManager.h"
#include "LocalFMSwitchInterface.h"

#include "fm_log.h"
#include <g_lwconfig.h>


/*****************************************************************************/
/*  This is a thread that gets spawned per-switch by Local Fabric Manager.   */      
/*  It operates on a vector of IOCTL pointers, which it issues serially to   */
/*  its instance of the switch driver.                                       */
/*****************************************************************************/

LocalFMSwitchInterface::LocalFMSwitchInterface( LWSWITCH_DEVICE_INSTANCE_INFO_V2 switchInfo,
                                                uint32_t switchHeartbeatTimeout )
{
    LW_STATUS retVal;

    mpLWSwitchDev = NULL;

    // update our local PCI BDF information
    mPciInfo.domain = switchInfo.pciDomain;
    mPciInfo.bus = switchInfo.pciBus;
    mPciInfo.device = switchInfo.pciDevice;
    mPciInfo.function = switchInfo.pciFunction;
    snprintf(mPciInfo.busId, FM_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
             FM_DEVICE_PCI_BUS_ID_FMT, FM_DEVICE_PCI_BUS_ID_FMT_ARGS(&mPciInfo));

    // update local uuid information. first clear our entire UUID memory area and copy only what shim layer returns.
    memset(mUuid.bytes, 0, FM_UUID_BUFFER_SIZE);
    lwswitch_uuid_to_string(&switchInfo.uuid, mUuid.bytes, FM_UUID_BUFFER_SIZE);

    // cache switch instance/enumeration index
    mSwitchInstanceId = switchInfo.deviceInstance;
    
    retVal = lwswitch_api_create_device(&switchInfo.uuid, &mpLWSwitchDev);
    if ( retVal != LW_OK )
    {
        std::ostringstream ss;
        ss << "request to open handle to LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId
           << " failed with error: " << lwstatusToString(retVal);        
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    //
    // before issuing any IOCTL to the driver, acquire our fabric management capability which indicate
    // fabric manager has enough permission to issue privileged IOCTLs. This is required to run FM
    // from root and non-root context. Also, capability is per switch device
    //

    // this function will throw exception if we fail to acquire the required capability
    acquireFabricManagementCapability();

    if ( !setSwitchHeartbeatKeepAliveTimeout(switchHeartbeatTimeout) )
    {
        std::ostringstream ss;
        ss << "unable to set heartbeat timeout configuration for LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId;
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    if ( !fetchSwitchArchInfo() )
    {
        std::ostringstream ss;
        ss << "failed to get architecture type information for LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId;
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    // update our gpio based switch physical id
    if ( !fetchSwitchPhysicalId() )
    {
        std::ostringstream ss;
        ss << "failed to get GPIO based physical id for LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId;
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    if ( !fetchSwitchPortInfo() )
    {
        std::ostringstream ss;
        ss << "failed to get number of port information for LWSwitch index:" << mSwitchInstanceId << " pci bus id:" << mPciInfo.busId;
        FM_LOG_ERROR("%s", ss.str().c_str());
        throw std::runtime_error(ss.str());
    }

    //
    // the switch driver keeps an error read position/index for each error type for
    // each client, such as Fabric Manager. Each client needs to maintain this index
    // to get the next error.
    //
    // At start, FM starts with index 0 for each error type, flushes all errors that
    // have oclwrred, and updates the index for each error type
    //
    // At runtime, index for each error type gets updated after each read.
    //
    for ( int errType = 0; errType < LWSWITCH_ERROR_SEVERITY_MAX; errType++ )
    {
        mSwitchErrorIndex[errType] = 0;
        flushSwitchError((LWSWITCH_ERROR_SEVERITY_TYPE)errType);
    }
}

LocalFMSwitchInterface::~LocalFMSwitchInterface()
{
    if ( mpLWSwitchDev )
    {
        lwswitch_api_free_device(&mpLWSwitchDev);
        mpLWSwitchDev = NULL;
    }
};

void LocalFMSwitchInterface::dumpIoctlGetInfo( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INFO *ioctlParams = (LWSWITCH_GET_INFO *)ioctl->ioctlParams;
    uint32_t i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("%5s %10s", "index", "info");

    for ( i = 0; i < ioctlParams->count; i++ )
    {
        FM_LOG_DEBUG("%5d 0x%08x %10d",
                    ioctlParams->index[i], ioctlParams->info[i], ioctlParams->info[i]);
    }
}

void LocalFMSwitchInterface::dumpIoctlSwitchPortConfig( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_SWITCH_PORT_CONFIG *ioctlParams =
            (LWSWITCH_SET_SWITCH_PORT_CONFIG *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, type %d, requesterLinkID %d, requesterLanID %d, count %d",
                 ioctlParams->portNum, ioctlParams->type, ioctlParams->requesterLinkID,
                 ioctlParams->requesterLanID, ioctlParams->count);
}

void LocalFMSwitchInterface::dumpIoctlGetIngressReqLinkID( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d", ioctlParams->portNum);

    FM_LOG_DEBUG("requesterLinkID %d", ioctlParams->requesterLinkID);
}


void LocalFMSwitchInterface::dumpIoctlGetIngressReq( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for (int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("idx %d, mappedAddress 0x%x, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].idx,
                     ioctlParams->entries[i].entry.mappedAddress,
                     ioctlParams->entries[i].entry.routePolicy,
                     ioctlParams->entries[i].entry.vcModeValid7_0,
                     ioctlParams->entries[i].entry.vcModeValid15_8,
                     ioctlParams->entries[i].entry.vcModeValid17_16,
                     ioctlParams->entries[i].entry.entryValid);
    }
}


void LocalFMSwitchInterface::dumpIoctlSetIngressReq( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_REQUEST_TABLE *ioctlParams =
            (LWSWITCH_SET_INGRESS_REQUEST_TABLE *)ioctl->ioctlParams;
    LWSWITCH_INGRESS_REQUEST_ENTRY  *entry;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("mappedAddress 0x%x, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].mappedAddress,
                     ioctlParams->entries[i].routePolicy,
                     ioctlParams->entries[i].vcModeValid7_0,
                     ioctlParams->entries[i].vcModeValid15_8,
                     ioctlParams->entries[i].vcModeValid17_16,
                     ioctlParams->entries[i].entryValid);
    }
}

void LocalFMSwitchInterface::dumpIoctlSetIngressValid( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_REQUEST_VALID *ioctlParams =
            (LWSWITCH_SET_INGRESS_REQUEST_VALID *)ioctl->ioctlParams;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("%d", ioctlParams->entryValid[i]);
    }
}

void LocalFMSwitchInterface::dumpIoctlGetIngressResp( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *ioctlParams =
            (LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for (int i = 0; i < (int)ioctlParams->numEntries; i++ )
    {

        FM_LOG_DEBUG("idx %d, routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].idx,
                     ioctlParams->entries[i].entry.routePolicy,
                     ioctlParams->entries[i].entry.vcModeValid7_0,
                     ioctlParams->entries[i].entry.vcModeValid15_8,
                     ioctlParams->entries[i].entry.vcModeValid17_16,
                     ioctlParams->entries[i].entry.entryValid);
    }
}

void LocalFMSwitchInterface::dumpIoctlSetIngressResp( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE *ioctlParams =
            (LWSWITCH_SET_INGRESS_RESPONSE_TABLE *)ioctl->ioctlParams;
    LWSWITCH_INGRESS_RESPONSE_ENTRY  *entry;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    for ( i = 0; i < (int)ioctlParams->numEntries; i++ )
    {

        FM_LOG_DEBUG("routePolicy %d,  vcModeValid7_0 0x%x, vcModeValid15_8 0x%x, vcModeValid17_16 0x%x, entryValid %d",
                     ioctlParams->entries[i].routePolicy,
                     ioctlParams->entries[i].vcModeValid7_0,
                     ioctlParams->entries[i].vcModeValid15_8,
                     ioctlParams->entries[i].vcModeValid17_16,
                     ioctlParams->entries[i].entryValid);
    }
}

void LocalFMSwitchInterface::dumpIoctlSetGangedLink( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_GANGED_LINK_TABLE *ioctlParams =
            (LWSWITCH_SET_GANGED_LINK_TABLE *)ioctl->ioctlParams;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("link_mask 0x%x", ioctlParams->link_mask);

    for ( i = 0; i < LWSWITCH_GANGED_LINK_TABLE_ENTRIES_MAX; i++ )
    {
        FM_LOG_DEBUG("entries[%d] 0x%x", i, ioctlParams->entries[i]);
    }
}

void LocalFMSwitchInterface::dumpIoctlSetLatencyBin( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_LATENCY_BINS *ioctlParams =
            (LWSWITCH_SET_LATENCY_BINS *)ioctl->ioctlParams;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("%2s %12s %12s %11s",
                "VC", "lowThreshold", "medThreshold", "hiThreshold");

    for ( i = 0; i < LWSWITCH_MAX_VCS; i++ )
    {
        FM_LOG_DEBUG("%2d %12d %12d %11d",
                i,
                ioctlParams->bin[i].lowThreshold,
                ioctlParams->bin[i].medThreshold,
                ioctlParams->bin[i].hiThreshold);
    }
}

void LocalFMSwitchInterface::dumpIoctlGetLwspliptCounterConfig( switchIoctl_t *ioctl )
{
     LWSWITCH_GET_LWLIPT_COUNTER_CONFIG *ioctlParams =
            (LWSWITCH_GET_LWLIPT_COUNTER_CONFIG *)ioctl->ioctlParams;
    int i;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void LocalFMSwitchInterface::dumpIoctlSetLwspliptCounterConfig( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_LWLIPT_COUNTER_CONFIG *ioctlParams =
            (LWSWITCH_SET_LWLIPT_COUNTER_CONFIG *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void LocalFMSwitchInterface::dumpIoctlGetErrors( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_ERRORS_PARAMS *ioctlParams = (LWSWITCH_GET_ERRORS_PARAMS *)ioctl->ioctlParams;
    uint32_t i;
    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
    FM_LOG_DEBUG("errorType 0x%08x, errorCount %d",
                ioctlParams->errorType, ioctlParams->errorCount);
    FM_LOG_DEBUG("%5s %3s %4s %7s %16s %8s",
                "value", "src", "inst", "subinst", "time", "resolved");
    for ( i = 0; i < ioctlParams->errorCount; i++ )
    {
        FM_LOG_DEBUG("%5d %3d %4d %7d %16llu %8d",
                    ioctlParams->error[i].error_value,
                    ioctlParams->error[i].error_src, ioctlParams->error[i].instance,
                    ioctlParams->error[i].subinstance, ioctlParams->error[i].time,
                    ioctlParams->error[i].error_resolved);
    }
}

void LocalFMSwitchInterface::dumpIoctlGetFatalErrorScope( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *ioctlParams = (LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("device %d", ioctlParams->device);
    for ( int i = 0; i < LWSWITCH_MAX_PORTS; i++ )
    {
        if ( ioctlParams->port[i] )
        {
            FM_LOG_DEBUG("port[%d] = %d", i, ioctlParams->port[i]);
        }
    }
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void LocalFMSwitchInterface::dumpIoctlGetFomValues( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_FOM_VALUES_PARAMS *ioctlParams = (LWSWITCH_GET_FOM_VALUES_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
    std::stringstream outStr;
    outStr << "FOM values LWLinkIndex:" << int(ioctlParams->linkId) << "numLanes:" << int(ioctlParams->numLanes) << "Values: ";
    for(int i = 0 ; i < LWSWITCH_LWLINK_MAX_LANES; i++)
        outStr << setw(3) << int(ioctlParams->figureOfMeritValues[i]);

    FM_LOG_DEBUG("%s", outStr.str().c_str());
}

void LocalFMSwitchInterface::dumpIoctlGetGradingValues( switchIoctl_t *ioctl )
{
    LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS *ioctlParams = (LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
    FM_LOG_DEBUG("LWLinkIndex:%d 0x%x", ioctlParams->linkId, ioctlParams->laneMask);
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++) {
        FM_LOG_DEBUG("Grading values tx_init[%d]:%u", i, ioctlParams->grading.tx_init[i]);
    }
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++) {
        FM_LOG_DEBUG("Grading values rx_init[%d]:%u", i, ioctlParams->grading.rx_init[i]);
    }
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++) {
        FM_LOG_DEBUG("Grading values tx_maint[%d]:%u", i, ioctlParams->grading.tx_maint[i]);
    }
    for(int i = 0; i < LWSWITCH_CCI_XVCR_LANES; i++) {
        FM_LOG_DEBUG("Grading values rx_maint[%d]:%u", i, ioctlParams->grading.rx_maint[i]);
    }
}
#endif

void LocalFMSwitchInterface::dumpIoctlUnregisterLink( switchIoctl_t *ioctl )
{
    LWSWITCH_UNREGISTER_LINK_PARAMS *ioctlParams = (LWSWITCH_UNREGISTER_LINK_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d", ioctlParams->portNum);
}

void LocalFMSwitchInterface::dumpIoctlResetAndDrainLinks( switchIoctl_t *ioctl )
{
    LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS *ioctlParams = (LWSWITCH_RESET_AND_DRAIN_LINKS_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("linkMask  0x%llx", ioctlParams->linkMask);
}

void LocalFMSwitchInterface::dumpIoctlExcludedDevice( switchIoctl_t *ioctl )
{
    LWSWITCH_BLACKLIST_DEVICE_PARAMS *ioctlParams = (LWSWITCH_BLACKLIST_DEVICE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void LocalFMSwitchInterface::dumpIoctlSetFmDriverFabricState( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_FM_DRIVER_STATE_PARAMS *ioctlParams = (LWSWITCH_SET_FM_DRIVER_STATE_PARAMS *)ioctl->ioctlParams;
    
    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void LocalFMSwitchInterface::dumpIoctlSetFmDeviceFabricState( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS *ioctlParams = (LWSWITCH_SET_DEVICE_FABRIC_STATE_PARAMS *)ioctl->ioctlParams;
    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void LocalFMSwitchInterface::dumpIoctlSetFmHeartbeatTimeout( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS *ioctlParams = (LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }
}

void
LocalFMSwitchInterface::dumpIoctlSetRemapPolicy( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_REMAP_POLICY *ioctlParams = (LWSWITCH_SET_REMAP_POLICY *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, tableSelect %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->tableSelect, ioctlParams->firstIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("entryValid %d, targetId %d, irlSelect %d, flags 0x%x, reqCtxMask 0x%x, Chk 0x%x, Rep 0x%x",
                    ioctlParams->remapPolicy[i].entryValid,
                    ioctlParams->remapPolicy[i].targetId,
                    ioctlParams->remapPolicy[i].irlSelect,
                    ioctlParams->remapPolicy[i].flags,
                    ioctlParams->remapPolicy[i].reqCtxMask,
                    ioctlParams->remapPolicy[i].reqCtxChk,
                    ioctlParams->remapPolicy[i].reqCtxRep);

        FM_LOG_DEBUG("address 0x%llx, Offset 0x%llx, Base 0x%llx, Limit 0x%llx",
                    ioctlParams->remapPolicy[i].address,
                    ioctlParams->remapPolicy[i].addressOffset,
                    ioctlParams->remapPolicy[i].addressBase,
                    ioctlParams->remapPolicy[i].addressLimit);
    }
}

void
LocalFMSwitchInterface::dumpIoctlGetRemapPolicy( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_REMAP_POLICY_PARAMS *ioctlParams = (LWSWITCH_GET_REMAP_POLICY_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, tableSelect %d, firstIndex %d, nextIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->tableSelect, ioctlParams->firstIndex,
                ioctlParams->nextIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_REMAP_POLICY_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("entryValid %d, targetId %d, irlSelect %d, flags 0x%x, reqCtxMask 0x%x, Chk 0x%x, Rep 0x%x",
                    ioctlParams->entry[i].entryValid,
                    ioctlParams->entry[i].targetId,
                    ioctlParams->entry[i].irlSelect,
                    ioctlParams->entry[i].flags,
                    ioctlParams->entry[i].reqCtxMask,
                    ioctlParams->entry[i].reqCtxChk,
                    ioctlParams->entry[i].reqCtxRep);

        FM_LOG_DEBUG("address 0x%llx, Offset 0x%llx, Base 0x%llx, Limit 0x%llx",
                    ioctlParams->entry[i].address,
                    ioctlParams->entry[i].addressOffset,
                    ioctlParams->entry[i].addressBase,
                    ioctlParams->entry[i].addressLimit);
    }
}

void
LocalFMSwitchInterface::dumpIoctlSetRoutingId( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_ROUTING_ID *ioctlParams = (LWSWITCH_SET_ROUTING_ID *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_ROUTING_ID_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_ROUTING_ID_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("index %d: entryValid %d, useRoutingLan %d, enableIrlErrResponse %d, numEntries %d",
                    ioctlParams->firstIndex + i,
                    ioctlParams->routingId[i].entryValid,
                    ioctlParams->routingId[i].useRoutingLan,
                    ioctlParams->routingId[i].enableIrlErrResponse,
                    ioctlParams->routingId[i].numEntries);

        if (ioctlParams->routingId[i].numEntries > LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX) {
            ioctlParams->routingId[i].numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
        }

        for ( uint j = 0; j < ioctlParams->routingId[i].numEntries; j++ )
        {
            FM_LOG_DEBUG("index %d: vcMap %d, destPortNum %d",
                        j, ioctlParams->routingId[i].portList[j].vcMap,
                        ioctlParams->routingId[i].portList[j].destPortNum);
        }
    }
}

void
LocalFMSwitchInterface::dumpIoctlGetRoutingId( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_ROUTING_ID_PARAMS *ioctlParams = (LWSWITCH_GET_ROUTING_ID_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, nextIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex,
                ioctlParams->nextIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_ROUTING_ID_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_ROUTING_ID_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("index %d: entryValid %d, useRoutingLan %d, enableIrlErrResponse %d, numEntries %d",
                    ioctlParams->entries[i].idx ,
                    ioctlParams->entries[i].entry.entryValid,
                    ioctlParams->entries[i].entry.useRoutingLan,
                    ioctlParams->entries[i].entry.enableIrlErrResponse,
                    ioctlParams->entries[i].entry.numEntries);

        if (ioctlParams->entries[i].entry.numEntries > LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX) {
            ioctlParams->entries[i].entry.numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
        }

        for ( uint j = 0; j < ioctlParams->entries[i].entry.numEntries; j++ )
        {
            FM_LOG_DEBUG("index %d: vcMap %d, destPortNum %d",
                        j, ioctlParams->entries[i].entry.portList[j].vcMap,
                        ioctlParams->entries[i].entry.portList[j].destPortNum);
        }
    }
}

void
LocalFMSwitchInterface::dumpIoctlSetRoutingLan( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_ROUTING_LAN *ioctlParams = (LWSWITCH_SET_ROUTING_LAN *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_ROUTING_LAN_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("index %d: entryValid %d, numEntries %d",
                    ioctlParams->firstIndex + i,
                    ioctlParams->routingLan[i].entryValid,
                    ioctlParams->routingLan[i].numEntries);

        if (ioctlParams->routingLan[i].numEntries > LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX) {
            ioctlParams->routingLan[i].numEntries = LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
        }

        for ( uint j = 0; j < ioctlParams->routingLan[i].numEntries; j++ )
        {
            FM_LOG_DEBUG("index %d: groupSelect %d, groupSize %d",
                        j, ioctlParams->routingLan[i].portList[j].groupSelect,
                        ioctlParams->routingLan[i].portList[j].groupSize);
        }
    }
}

void
LocalFMSwitchInterface::dumpIoctlGetRoutingLan( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_ROUTING_LAN_PARAMS *ioctlParams = (LWSWITCH_GET_ROUTING_LAN_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("portNum %d, firstIndex %d, nextIndex %d, numEntries %d",
                ioctlParams->portNum, ioctlParams->firstIndex,
                ioctlParams->nextIndex, ioctlParams->numEntries);

    if (ioctlParams->numEntries > LWSWITCH_ROUTING_LAN_ENTRIES_MAX) {
        ioctlParams->numEntries = LWSWITCH_ROUTING_LAN_ENTRIES_MAX;
    }

    for ( uint i = 0; i < ioctlParams->numEntries; i++ )
    {
        FM_LOG_DEBUG("index %d: entryValid %d, numEntries %d",
                    ioctlParams->entries[i].idx,
                    ioctlParams->entries[i].entry.entryValid,
                    ioctlParams->entries[i].entry.numEntries);

        if (ioctlParams->entries[i].entry.numEntries > LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX) {
            ioctlParams->entries[i].entry.numEntries = LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX;
        }

        for ( uint j = 0; j < ioctlParams->entries[i].entry.numEntries; j++ )
        {
            FM_LOG_DEBUG("index %d: groupSelect %d, groupSize %d",
                        j, ioctlParams->entries[i].entry.portList[j].groupSelect,
                        ioctlParams->entries[i].entry.portList[j].groupSize);
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
void
LocalFMSwitchInterface::dumpIoctlSetMcRidTable( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *ioctlParams = (LWSWITCH_SET_MC_RID_TABLE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("physicalId 0x%x, portNum %d, index %d, extendedTable %d, entryValid %d, mcSize %d, numSprayGroups %d, "
                 "extendedPtr %d, extendedValid %d, noDynRsp %d",
                 mPhysicalId, ioctlParams->portNum, ioctlParams->index, ioctlParams->extendedTable,
                 ioctlParams->entryValid, ioctlParams->mcSize, ioctlParams->numSprayGroups,
                 ioctlParams->extendedPtr, ioctlParams->extendedValid, ioctlParams->noDynRsp);

    uint32_t portIndex = 0;
    uint32_t numSprayGroups = ioctlParams->numSprayGroups;

    if (numSprayGroups > LWSWITCH_MC_MAX_SPRAYGROUPS) {
        numSprayGroups = LWSWITCH_MC_MAX_SPRAYGROUPS;
    }

    for (uint32_t i = 0; i < numSprayGroups; i++) {
        FM_LOG_DEBUG("spray %d, replicaOffset %d, replicaValid %d",
                     i, ioctlParams->replicaOffset[i], ioctlParams->replicaValid[i]);

        uint32_t portsPerSprayGroup = ioctlParams->portsPerSprayGroup[i];
        if (portsPerSprayGroup > LWSWITCH_MC_MAX_PORTS) {
            portsPerSprayGroup = LWSWITCH_MC_MAX_PORTS;
        }

        for (uint32_t j = 0; j < portsPerSprayGroup; j++, portIndex++) {
            FM_LOG_DEBUG("%d: (%d, %d) ",
                         portIndex, ioctlParams->ports[portIndex], ioctlParams->vcHop[portIndex]);
        }
    }
}

void
LocalFMSwitchInterface::dumpIoctlGetMcRidTable( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *ioctlParams = (LWSWITCH_GET_MC_RID_TABLE_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("physicalId 0x%x, portNum %d, index %d, extendedTable %d, entryValid %d, mcSize %d, numSprayGroups %d, "
                 "extendedPtr %d, extendedValid %d, noDynRsp %d",
                 mPhysicalId, ioctlParams->portNum, ioctlParams->index, ioctlParams->extendedTable,
                 ioctlParams->entryValid, ioctlParams->mcSize, ioctlParams->numSprayGroups,
                 ioctlParams->extendedPtr, ioctlParams->extendedValid, ioctlParams->noDynRsp);

    uint32_t portIndex = 0;
    uint32_t numSprayGroups = ioctlParams->numSprayGroups;

    if (numSprayGroups > LWSWITCH_MC_MAX_SPRAYGROUPS) {
        numSprayGroups = LWSWITCH_MC_MAX_SPRAYGROUPS;
    }

    for (uint32_t i = 0; i < numSprayGroups; i++) {
        FM_LOG_DEBUG("spray %d, replicaOffset %d, replicaValid %d",
                     i, ioctlParams->replicaOffset[i], ioctlParams->replicaValid[i]);

        uint32_t portsPerSprayGroup = ioctlParams->portsPerSprayGroup[i];
        if (portsPerSprayGroup > LWSWITCH_MC_MAX_PORTS) {
            portsPerSprayGroup = LWSWITCH_MC_MAX_PORTS;
        }

        for (uint32_t j = 0; j < portsPerSprayGroup; j++, portIndex++) {
            FM_LOG_DEBUG("%d: (%d, %d) ",
                         portIndex, ioctlParams->ports[portIndex], ioctlParams->vcHop[portIndex]);
        }
    }
}
#endif

void
LocalFMSwitchInterface::dumpIoctlTrainingErrorInfo( switchIoctl_t *ioctl )
{
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *ioctlParams = (LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("attemptedTrainingMask0 0x%llx trainingErrorMask0 0x%llx",
                 ioctlParams->attemptedTrainingMask0, ioctlParams->trainingErrorMask0);
}

void LocalFMSwitchInterface::dumpIoctlGetLwliptCounters( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_LWLIPT_COUNTERS *ioctlParams = (LWSWITCH_GET_LWLIPT_COUNTERS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("%4s  %16s  %16s %16s %16s",
                 "port", "txCounter0", "txCounter1", "rxCounter0", "rxCounter1");

    for (int portNum = 0; portNum < LWSWITCH_MAX_PORTS; portNum++ ) {
        FM_LOG_DEBUG("%4d  %16llu  %16llu %16llu %16llu",
                     portNum, ioctlParams->liptCounter[portNum].txCounter0,
                     ioctlParams->liptCounter[portNum].txCounter1,
                     ioctlParams->liptCounter[portNum].rxCounter0,
                     ioctlParams->liptCounter[portNum].rxCounter1);
    }
}

void LocalFMSwitchInterface::dumpIoctlGetThrouputCounters( switchIoctl_t *ioctl )
{
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *ioctlParams = (LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *)ioctl->ioctlParams;

    if ( !ioctlParams )
    {
        FM_LOG_DEBUG("Invalid params");
        return;
    }

    FM_LOG_DEBUG("%4s  %16s  %16s %16s %16s",
                 "port", "data_tx", "data_rx", "raw_tx", "raw_rx");

    for (int portNum = 0; portNum < LWSWITCH_MAX_PORTS; portNum++ ) {
        FM_LOG_DEBUG("%4d  %16llu  %16llu %16llu %16llu",
                     portNum, ioctlParams->counters[portNum].values[0],
                     ioctlParams->counters[portNum].values[1],
                     ioctlParams->counters[portNum].values[2],
                     ioctlParams->counters[portNum].values[3]);
    }
}

void LocalFMSwitchInterface::dumpIoctl( switchIoctl_t *ioctl )
{
    if ( !ioctl )
    {
        FM_LOG_DEBUG("Invalid ioctl");
        return;
    }

    switch ( ioctl->type )
    {
    case IOCTL_LWSWITCH_GET_INFO:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_INFO(0x%x)", ioctl->type);
        dumpIoctlGetInfo( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG(0x%x)", ioctl->type);
        dumpIoctlSwitchPortConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE(0x%x)", ioctl->type);
        dumpIoctlSetIngressReq( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE(0x%x)", ioctl->type);
        dumpIoctlGetIngressReq( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID(0x%x)", ioctl->type);
        dumpIoctlSetIngressValid( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_REQLINKID:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_INGRESS_REQLINKID(0x%x)", ioctl->type);
        dumpIoctlGetIngressReqLinkID( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE(0x%x)", ioctl->type);
        dumpIoctlGetIngressResp( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE(0x%x)", ioctl->type);
        dumpIoctlSetIngressResp( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE(0x%x)", ioctl->type);
        dumpIoctlSetGangedLink( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_LATENCY_BINS:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_LATENCY_BINS(0x%x)", ioctl->type);
        dumpIoctlSetLatencyBin( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG(0x%x)", ioctl->type);
        dumpIoctlGetLwspliptCounterConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG(0x%x)", ioctl->type);
        dumpIoctlSetLwspliptCounterConfig( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_ERRORS:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_ERRORS(0x%x)", ioctl->type);
        dumpIoctlGetErrors( ioctl );
        break;

    case IOCTL_LWSWITCH_UNREGISTER_LINK:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_UNREGISTER_LINK(0x%x)", ioctl->type);
        dumpIoctlUnregisterLink( ioctl );
        break;

    case IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS:
         FM_LOG_DEBUG("IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS(0x%x)", ioctl->type);
         dumpIoctlResetAndDrainLinks( ioctl );
         break;

    case IOCTL_LWSWITCH_BLACKLIST_DEVICE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_BLACKLIST_DEVICE(0x%x)", ioctl->type);
        dumpIoctlExcludedDevice( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_FM_DRIVER_STATE:
        //FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_FM_DRIVER_STATE(0x%x)", ioctl->type);
        dumpIoctlSetFmDriverFabricState( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE(0x%x)", ioctl->type);
        dumpIoctlSetFmDeviceFabricState( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT(0x%x)", ioctl->type);
        dumpIoctlSetFmHeartbeatTimeout( ioctl );
        break; 

    case IOCTL_LWSWITCH_SET_REMAP_POLICY:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_REMAP_POLICY(0x%x)", ioctl->type);
        dumpIoctlSetRemapPolicy( ioctl );
        break;

    case IOCTL_LWSWITCH_SET_ROUTING_ID:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_ROUTING_ID(0x%x)", ioctl->type);
        dumpIoctlSetRoutingId( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_REMAP_POLICY:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_REMAP_POLICY(0x%x)", ioctl->type);
        dumpIoctlGetRemapPolicy( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_ROUTING_ID:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_ROUTING_ID(0x%x)", ioctl->type);
        dumpIoctlGetRoutingId( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_ROUTING_LAN:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_ROUTING_LAN(0x%x)", ioctl->type);
        dumpIoctlGetRoutingLan( ioctl );
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case IOCTL_LWSWITCH_SET_MC_RID_TABLE:
        FM_LOG_INFO("IOCTL_LWSWITCH_SET_MC_RID_TABLE(0x%x)", ioctl->type);
        dumpIoctlSetMcRidTable( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_MC_RID_TABLE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_MC_RID_TABLE(0x%x)", ioctl->type);
        dumpIoctlGetMcRidTable( ioctl );
        break;

#endif

    case IOCTL_LWSWITCH_SET_TRAINING_ERROR_INFO:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_SET_TRAINING_ERROR_INFO(0x%x)", ioctl->type);
        dumpIoctlTrainingErrorInfo( ioctl );
        break;

    case IOCTL_LWSWITCH_GET_FATAL_ERROR_SCOPE:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_FATAL_ERROR_SCOPE(0x%x)", ioctl->type);
        dumpIoctlGetFatalErrorScope( ioctl );
        break;

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    case IOCTL_LWSWITCH_GET_FOM_VALUES:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_GET_FOM_VALUES(0x%x)", ioctl->type);
        dumpIoctlGetFomValues( ioctl );
        break;

    case IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES:
        FM_LOG_DEBUG("IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES(0x%x)", ioctl->type);
        dumpIoctlGetGradingValues( ioctl );
        break;
#endif
    default:
        FM_LOG_DEBUG("Unknown ioctl->type 0x%x", ioctl->type);
        break;
    }
};

FMIntReturn_t
LocalFMSwitchInterface::doIoctl( switchIoctl_t *pIoctl )
{
    LW_STATUS retVal;

    switch ( pIoctl->type )
    {
    case IOCTL_LWSWITCH_GET_INFO:
    case IOCTL_LWSWITCH_SET_SWITCH_PORT_CONFIG:
    case IOCTL_LWSWITCH_GET_INGRESS_REQUEST_TABLE:
    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_TABLE:
    case IOCTL_LWSWITCH_SET_INGRESS_REQUEST_VALID:
    case IOCTL_LWSWITCH_GET_INGRESS_REQLINKID:
    case IOCTL_LWSWITCH_GET_INGRESS_RESPONSE_TABLE:
    case IOCTL_LWSWITCH_SET_INGRESS_RESPONSE_TABLE:
    case IOCTL_LWSWITCH_SET_GANGED_LINK_TABLE:
    case IOCTL_LWSWITCH_SET_LATENCY_BINS:
    case IOCTL_LWSWITCH_GET_LWLIPT_COUNTER_CONFIG:
    case IOCTL_LWSWITCH_SET_LWLIPT_COUNTER_CONFIG:
    case IOCTL_LWSWITCH_GET_LWLINK_COUNTERS:
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    case IOCTL_LWSWITCH_GET_FOM_VALUES:
    case IOCTL_LWSWITCH_CCI_GET_GRADING_VALUES:
#endif
    // LwSwitch Error Data Control and Collection
    case IOCTL_LWSWITCH_GET_ERRORS:
    case IOCTL_LWSWITCH_GET_FATAL_ERROR_SCOPE:
    case IOCTL_LWSWITCH_UNREGISTER_LINK:
    case IOCTL_LWSWITCH_RESET_AND_DRAIN_LINKS:
    case IOCTL_LWSWITCH_BLACKLIST_DEVICE:
    case IOCTL_LWSWITCH_SET_FM_DRIVER_STATE:
    case IOCTL_LWSWITCH_SET_DEVICE_FABRIC_STATE:
    case IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT:
    // routing table
    case IOCTL_LWSWITCH_SET_REMAP_POLICY:
    case IOCTL_LWSWITCH_SET_ROUTING_ID:
    case IOCTL_LWSWITCH_SET_ROUTING_LAN:
    case IOCTL_LWSWITCH_GET_REMAP_POLICY:
    case IOCTL_LWSWITCH_GET_ROUTING_ID:
    case IOCTL_LWSWITCH_GET_ROUTING_LAN:
    case IOCTL_LWSWITCH_SET_TRAINING_ERROR_INFO:
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case IOCTL_LWSWITCH_SET_MC_RID_TABLE:
    case IOCTL_LWSWITCH_GET_MC_RID_TABLE:
        break;
#endif

    default:
        FM_LOG_ERROR("LWSwitch driver ioctl type 0x%x is not supported", pIoctl->type);
        return FM_INT_ST_IOCTL_ERR;
    }
    
    //FM_LOG_DEBUG("start cmd=0x%x", pIoctl->type);
    retVal = lwswitch_api_control( mpLWSwitchDev, pIoctl->type, pIoctl->ioctlParams, pIoctl->paramSize );
    //FM_LOG_DEBUG("end cmd=0x%x", pIoctl->type);

    if ( retVal != LW_OK ) {
        
        FM_LOG_ERROR( "LWSwitch driver ioctl type 0x%x failed for device index %d physical id %d pci bus id %s with error %s",
                      pIoctl->type, mSwitchInstanceId, mPhysicalId, mPciInfo.busId, lwstatusToString(retVal) );
        dumpIoctl( pIoctl );
        return FM_INT_ST_IOCTL_ERR;
    }

    dumpIoctl( pIoctl );
    return FM_INT_ST_OK;
}

uint32_t
LocalFMSwitchInterface::getSwitchDevIndex()
{
    return mSwitchInstanceId;
}

uint32_t
LocalFMSwitchInterface::getSwitchPhysicalId()
{
    return mPhysicalId;
}

const FMPciInfo_t&
LocalFMSwitchInterface::getSwtichPciInfo()
{
    return mPciInfo;
}

uint32_t
LocalFMSwitchInterface::getNumPorts()
{
    return mNumPorts;
}

uint64_t
LocalFMSwitchInterface::getEnabledPortMask()
{
    return mEnabledPortMask;
}

uint32_t
LocalFMSwitchInterface::getSwitchArchType()
{
    return mArchType;
}

const FMUuid_t&
LocalFMSwitchInterface::getUuid()
{
    return mUuid;
}

/* update cached enabled port mask information after unregistering/disabling ports */
void
LocalFMSwitchInterface::updateEnabledPortMask()
{
    fetchSwitchPortInfo();
}

bool 
LocalFMSwitchInterface::setSwitchHeartbeatKeepAliveTimeout(uint32_t switchHeartbeatTimeout)
{
    FMIntReturn_t rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT_PARAMS ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    //timeout in milliseconds
    ioctlParams.fmTimeout = switchHeartbeatTimeout;
    ioctlStruct.type = IOCTL_LWSWITCH_SET_FM_HEARTBEAT_TIMEOUT;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    rc = doIoctl( &ioctlStruct );

    if (rc != FM_INT_ST_OK)
    {
        return false;
    }

    return true;
}

bool
LocalFMSwitchInterface::fetchSwitchPhysicalId()
{
    FMIntReturn_t rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.count = 1;
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_INT_ST_OK )
    {
        // failed to get the switch physical id information
        return false;
    } 

    // update our local switch physical id information
    mPhysicalId = ioctlParams.info[0];
    return true;
}

bool
LocalFMSwitchInterface::fetchSwitchPortInfo()
{
    FMIntReturn_t rc;
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
    ioctlStruct.paramSize = sizeof(ioctlParams);

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_INT_ST_OK )
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
LocalFMSwitchInterface::fetchSwitchArchInfo()
{
    FMIntReturn_t rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_INFO ioctlParams;

    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.index[0] = LWSWITCH_GET_INFO_INDEX_ARCH;
    ioctlParams.count = 1;
    ioctlStruct.type = IOCTL_LWSWITCH_GET_INFO;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);

    rc = doIoctl( &ioctlStruct );
    if ( rc != FM_INT_ST_OK )
    {
        // failed to get the switch port config items
        return false;
    }

    // Get the switch arch
    mArchType = ioctlParams.info[0];
    return true;
}

void
LocalFMSwitchInterface::acquireFabricManagementCapability()
{
    LW_STATUS retVal;

    //
    // by default all the switch device nodes (/dev/lwpu-lwswitchX in Linux ) has permission
    // to all users. All the privileged access is controlled through special fabric management node.
    // The actual management node dependents on driver mechanism. For devfs based support, it will be
    // /dev/lwpu-caps/lwpu-capX and for procfs based, it will be /proc/driver/lwpu-lwlink/capabilities/fabric-mgmt
    // This entry is created by driver and default access is for root/admin. The system administrator then must
    // change access to desired user. The below API is verifying whether FM has access to the path, if so open it
    // and associate/link the corresponding file descriptor with the file descriptor associated with the
    // current device node file descriptor (ie fd of /dev/lwpu-lwswitchX)
    //

    retVal = lwswitch_api_acquire_capability(mpLWSwitchDev, LWSWITCH_CAP_FABRIC_MANAGEMENT);
    if ( retVal != LW_OK )
    {
        // failed to get capability. throw error based on common return values
        switch (retVal)
        {
            case LW_ERR_INSUFFICIENT_PERMISSIONS:
            {
                std::ostringstream ss;
                ss << "failed to acquire required privileges to access LWSwitch devices." <<
                      " make sure fabric manager has access permissions to required device node files";
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
                break;
            }
            case LW_ERR_NOT_SUPPORTED:
            {
                //
                // driver doesn't have fabric management capability support on Windows for now and will 
                // return LW_ERR_NOT_SUPPORTED. So, treat this as not an error for now to let Windows FM to
                // continue. In Linux, the assumption is that Driver will not return LW_ERR_NOT_SUPPORTED
                // and even if, FM will eventually fail as privileged control calls will start erroring out.
                //
                break;
            }
            default:
            {
                std::ostringstream ss;
                ss << "request to acquire required privileges to access LWSwitch devices failed with error:" << lwstatusToString(retVal);
                FM_LOG_ERROR("%s", ss.str().c_str());
                throw std::runtime_error(ss.str());
                break;
            }
        }
    }

    // successfully acquired required fabric management capability
}

uint64_t
LocalFMSwitchInterface::getErrorIndex(LWSWITCH_ERROR_SEVERITY_TYPE errorType)
{
    return mSwitchErrorIndex[errorType];
}
void
LocalFMSwitchInterface::setErrorIndex(LWSWITCH_ERROR_SEVERITY_TYPE errorType, uint64_t errorIndex)
{
    mSwitchErrorIndex[errorType] = errorIndex;
}
/*
 * Flush switch errors from the switch interface and update the errorIndex for errorType
 *
 */
void
LocalFMSwitchInterface::flushSwitchError(LWSWITCH_ERROR_SEVERITY_TYPE errorType)
{
    FMIntReturn_t rc;
    switchIoctl_t ioctlStruct;
    LWSWITCH_GET_ERRORS_PARAMS ioctlParams;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlStruct.type = IOCTL_LWSWITCH_GET_ERRORS;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);
    if (doIoctl( &ioctlStruct ) != FM_INT_ST_OK)
    {
        FM_LOG_ERROR("failed to flush error type %d on LWSwitch index %d physical id %d pci bus id %s",
                     errorType, mSwitchInstanceId, mPhysicalId, mPciInfo.busId);
        return;
    }
    // update the errorIndex
    if (ioctlParams.errorCount > 0)
    {
        setErrorIndex(errorType, ioctlParams.nextErrorIndex);
    }
}
/*
 * Get switch errors from the switch interface and update the errorIndex for errorType
 */
FMIntReturn_t
LocalFMSwitchInterface::getSwitchErrors(LWSWITCH_ERROR_SEVERITY_TYPE errorType,
                                        LWSWITCH_GET_ERRORS_PARAMS &ioctlParams)
{
    FMIntReturn_t rc;
    switchIoctl_t ioctlStruct;
    memset(&ioctlParams, 0, sizeof(ioctlParams));
    ioctlParams.errorType = errorType;
    // set up the index to get the next error
    ioctlParams.errorIndex = getErrorIndex(errorType);
    ioctlStruct.type = IOCTL_LWSWITCH_GET_ERRORS;
    ioctlStruct.ioctlParams = &ioctlParams;
    ioctlStruct.paramSize = sizeof(ioctlParams);
    rc = doIoctl( &ioctlStruct );
    if (rc != FM_INT_ST_OK)
    {
        FM_LOG_ERROR("failed to get error type %d on LWSwitch index %d physical id %d pci bus id %s",
                     errorType, mSwitchInstanceId, mPhysicalId, mPciInfo.busId);
        return rc;
    }
    // update the errorIndex
    if (ioctlParams.errorCount > 0)
    {
        setErrorIndex(errorType, ioctlParams.nextErrorIndex);
    }
    return FM_INT_ST_OK;
}


