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
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>

#include "fm_log.h"
#include "GlobalFmFabricParser.h"
#include "FMDeviceProperty.h"
#include <g_lwconfig.h>
#include "lwmisc.h"
#include "lw_fm_types.h"

using namespace std;

#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
#include "modsdrv.h"
#include <streambuf>
#include <istream>

namespace
{
    class MemBuf : public basic_streambuf<char>
    {
        public:
            MemBuf(char *p, size_t l)
            {
                setg(p, p, p + l);
            }
    };
    class IMemStream : public istream
    {
        public:
            IMemStream(char *p, size_t l)
              : std::istream(&mBuffer),
                mBuffer(p, l)
            {
                rdbuf(&mBuffer);
            }

        private:
            MemBuf mBuffer;
    };

    FMIntReturn_t
    ReadTarFile(const char* topoFile, vector<char>* data)
    {
        uint32_t size = 0;
        if (0 == ModsDrvReadTarFile("lwlinktopofiles.INTERNAL.bin", topoFile, &size, nullptr))
        {
            data->resize(size);
            if (0 == ModsDrvReadTarFile("lwlinktopofiles.INTERNAL.bin", topoFile, &size, &(*data)[0]))
                return FM_INT_ST_OK;
        }

        if (0 == ModsDrvReadTarFile("lwlinktopofiles.bin", topoFile, &size, nullptr))
        {
            data->resize(size);
            if (0 == ModsDrvReadTarFile("lwlinktopofiles.bin", topoFile, &size, &(*data)[0]))
                return FM_INT_ST_OK;
        }

        return FM_INT_ST_FILE_OPEN_ERR;
    }
};
#endif

FMFabricParser::FMFabricParser( const char *fabricPartitionFileName )
{
    mpFabric = new( fabric );
    mpFabricCopy = NULL;
    lwLinkConnMap.clear();
    mSwitchTrunkPortMaskInfo.clear();
    mLoopbackPorts.clear();
    mDisabledPartitions.clear();
    mDisabledGpuEndpointIds.clear();
    mDisabledSwitches.clear();
    mIlwalidIngressReqEntries.clear();
    mIlwalidIngressRespEntries.clear();
    mGpuTargetIdTbl.clear();

    mIlwalidIngressRmapEntries.clear();
    mIlwalidIngressRidEntries.clear();
    mIlwalidIngressRlanEntries.clear();

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    mIlwalidIngressRmapExtAEntries.clear();
    mIlwalidIngressRmapExtBEntries.clear();
    mIlwalidIngressRmapMcEntries.clear();
#endif

    mFabricPartitionFileName = fabricPartitionFileName;
    mpPartitionInfo = NULL;
    sharedLwswitchPartitionCfg.clear();
    portsToReachGpuMap.clear();
    gpusReachableFromPortMap.clear();
};

FMFabricParser::~FMFabricParser()
{
    lwLinkConnMap.clear();

    fabricParserCleanup();

    if ( mpFabric )
    {
        delete( mpFabric );
        mpFabric = NULL;
    }

    if ( mpFabricCopy ) {
        delete( mpFabricCopy );
        mpFabricCopy = NULL;
    }

    if ( mpPartitionInfo )
    {
        delete( mpPartitionInfo );
        mpPartitionInfo = NULL;
    }

};

FMIntReturn_t
FMFabricParser::parseOneGpu( const GPU &gpu, uint32_t nodeId, int gpuIndex )
{
    lwswitch::gpuInfo *info;
    peerIDPortMap *pPeerIdMap;
    uint64_t addrBase, addrRange;
    GpuKeyType gpuKey;
    int physicalId = 0, gpuEndPointId = 0, targetId = 0;

    // find the gpu PhysicalId from fabric address.
    if ( gpu.has_fabricaddrbase() )
    {
        addrBase = gpu.fabricaddrbase();

        if ( gpu.has_physicalid() )
        {
            physicalId = gpu.physicalid();
        }
        else
        {
            FM_LOG_ERROR("missing physical Id for " NODE_ID_LOG_STR " %d GPU index %d in topology file", nodeId, gpuIndex);
            // In older topology file, there might be no GPU physicalId specified.
            // The base address of a Volta GPU is assigned as gpuEndPointId << 36, so do a reverse callwlation modulo the MAX number of SV10 GPUs
            // Keep the logic here to support those topology, this logic does not apply to Ampere and GPUs come
            // after Ampere.
            gpuEndPointId = (uint64_t)addrBase >> 36;
            physicalId = gpuEndPointId % 16;
        }
    }
    else
    {
        FM_LOG_ERROR("missing fabric address base for " NODE_ID_LOG_STR " %d GPU index %d in topology file", nodeId, gpuIndex);
        return FM_INT_ST_ILWALID_GPU_CFG;
    }

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, gpuIndex %d physicalId %d ",
                 nodeId, gpuIndex, physicalId);

    info = new lwswitch::gpuInfo();
    gpuKey.nodeId = nodeId;
    gpuKey.physicalId  = physicalId;
    gpuCfg.insert(make_pair(gpuKey, info));

    info->set_gpuphysicalid( physicalId );

    if ( gpu.has_targetid() )
    {
        targetId = gpu.targetid();
        info->set_targetid( targetId );
        mGpuTargetIdTbl.insert(make_pair(targetId, gpuKey));
    }

    if ( gpu.has_fabricaddrrange() )
    {
        addrRange = gpu.fabricaddrrange();
    }
    else
    {
        FM_LOG_ERROR("missing fabric address range for " NODE_ID_LOG_STR " %d GPU index %d in topology file", nodeId, gpuIndex);
        return FM_INT_ST_ILWALID_GPU_CFG;
    }

    if ( gpu.has_gpabase() && gpu.has_gparange() )
    {
        info->set_gpaaddressbase(gpu.gpabase());
        info->set_gpaaddressrange(gpu.gparange());
    }

    if ( gpu.has_flabase() && gpu.has_flarange() )
    {
        info->set_flaaddressbase(gpu.flabase());
        info->set_flaaddressrange(gpu.flarange());
    }

    if ( gpu.has_logicaltophyportmap() )
    {
        info->set_logicaltophyportmap( gpu.logicaltophyportmap() );
    }

    info->set_fabricaddressrange( addrRange );
    info->set_fabricaddressbase( addrBase );

    for ( int i = 0; i < gpu.peertoport_size(); i++ )
    {
        pPeerIdMap = info->add_map();
        pPeerIdMap->set_version( gpu.peertoport(i).version() );
        if ( gpu.peertoport(i).has_peerid() )
        {
            pPeerIdMap->set_peerid( gpu.peertoport(i).peerid() );
        }

        if ( gpu.peertoport(i).portmap_size() > 0 )
        {
            // TODO: handle GPU peerIdMap
            // pPeerIdMap->set_portmap( gpu.peertoport(i).portmap() );
        }
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                         uint32_t localPortNum,
                                         const ingressRequestTable &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    ingressRequestTable *entry;
    ReqTableKeyType key;

    entry = new ingressRequestTable;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    reqEntry.insert(make_pair(key, entry));

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                          uint32_t localPortNum,
                                          const ingressResponseTable &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    ingressResponseTable *entry;
    RespTableKeyType key;

    entry = new ingressResponseTable;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    respEntry.insert(make_pair(key, entry));

    return FM_INT_ST_OK;
}

// new routing tables introduced with LimeRock

FMIntReturn_t
FMFabricParser::parseOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                          RemapTable remapTable, const rmapPolicyEntry &cfg )
{
    int index = cfg.has_index() ? cfg.index() : -1;
    rmapPolicyEntry *entry;
    RmapTableKeyType key;

    entry = new rmapPolicyEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;

    if ( remapTable == NORMAL_RANGE )
    {
        rmapEntry.insert(make_pair(key, entry));
    }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    else if ( remapTable == EXTENDED_RANGE_A )
    {
        rmapExtAEntry.insert(make_pair(key, entry));
    }
    else if ( remapTable == EXTENDED_RANGE_B )
    {
        rmapExtBEntry.insert(make_pair(key, entry));
    }
#endif

    return FM_INT_ST_OK;
}

// construct ports to reach a GPU from the RID entry egress port list
void
FMFabricParser::constructPortsToReachGpuMap(uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum, const ridRouteEntry &cfg)
{
    if ( !cfg.has_index() || !cfg.has_valid() || !cfg.valid() )
    {
        return;
    }

    // RID tabled index is the targetId
    uint32_t targetId = (uint32_t)cfg.index();

    // construct GPU reachablity map from the RID entry
    SwitchKeyType switchKey;
    switchKey.nodeId  = nodeId;
    switchKey.physicalId = physicalId;

    PortsToReachGpuMap::iterator it = portsToReachGpuMap.find(switchKey);
    if ( it == portsToReachGpuMap.end() )
    {
        PortsToReachGpu portsToReachGpu;
        portsToReachGpu.clear();
        portsToReachGpuMap.insert( make_pair(switchKey, portsToReachGpu) );
        it = portsToReachGpuMap.find(switchKey);
    }

    PortsToReachGpu &portsToReachGpu = it->second;
    for (int i = 0; i < cfg.portlist_size(); i++ )
    {
        routePortList port = cfg.portlist(i);
        uint32_t portNum = port.portindex();

        PortsToReachGpu::iterator jit = portsToReachGpu.find( targetId );
        if ( jit == portsToReachGpu.end() )
        {
            portsToReachGpu.insert( make_pair(targetId, 0) );
            jit = portsToReachGpu.find( targetId );
        }
        uint64_t portMask = jit->second;
        portMask |= (uint64_t)1 << portNum;
        portsToReachGpu[targetId] = portMask;
    }
}

// construct GPUs that are reachable from a port from the RID entry egress port list
void
FMFabricParser::consructGpusReachableFromPortMap(uint32_t nodeId, uint32_t physicalId,
                                                 uint32_t localPortNum, const ridRouteEntry &cfg)
{
    if ( !cfg.has_index() || !cfg.has_valid() || !cfg.valid() )
    {
        return;
    }

    // RID tabled index is the targetId
    uint32_t targetId = (uint32_t)cfg.index();

    // construct GPU reachablity map from the RID entry
    SwitchKeyType switchKey;
    switchKey.nodeId  = nodeId;
    switchKey.physicalId = physicalId;

    GpusReachableFromPortMap::iterator it = gpusReachableFromPortMap.find(switchKey);
    if ( it == gpusReachableFromPortMap.end() )
    {
        GpusReachableFromPort gpusReachableFromPort;
        gpusReachableFromPort.clear();
        gpusReachableFromPortMap.insert( make_pair(switchKey, gpusReachableFromPort) );
        it = gpusReachableFromPortMap.find(switchKey);
    }

    GpusReachableFromPort &gpusReachableFromPort = it->second;
    for (int i = 0; i < cfg.portlist_size(); i++ )
    {
        routePortList port = cfg.portlist(i);
        uint32_t portNum = port.portindex();

        GpusReachableFromPort::iterator jit = gpusReachableFromPort.find( portNum );
        if ( jit == gpusReachableFromPort.end() )
        {
            GpuTargetIds gpus;
            gpus.clear();
            gpusReachableFromPort.insert( make_pair(portNum, gpus) );
            jit = gpusReachableFromPort.find( portNum );
        }

        GpuTargetIds &gpus = jit->second;
        gpus.insert( targetId );
    }
}

FMIntReturn_t
FMFabricParser::parseOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId,
                                         uint32_t localPortNum,
                                         const ridRouteEntry &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    ridRouteEntry *entry;
    RidTableKeyType key;

    entry = new ridRouteEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    ridEntry.insert(make_pair(key, entry));

    // construct GPU reachability maps
    constructPortsToReachGpuMap(nodeId, physicalId, localPortNum, cfg);
    consructGpusReachableFromPortMap(nodeId, physicalId, localPortNum, cfg);

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId,
                                          uint32_t localPortNum,
                                          const rlanRouteEntry &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    rlanRouteEntry *entry;
    RlanTableKeyType key;

    entry = new rlanRouteEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    rlanEntry.insert(make_pair(key, entry));

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneGangedLinkEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                         const gangedLinkTable &cfg )
{
    int32_t index, *data;
    GangedLinkTableKeyType key;

    if ( !IS_PORT_VALID(localPortNum) )
    {
        FM_LOG_ERROR("invalid local port number for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d",
                     nodeId, physicalId, localPortNum);
        return FM_INT_ST_ILWALID_TABLE_ENTRY;
    }

    for ( index = 0; index < cfg.data_size(); index++ )
    {
        data = new int32_t;
        *data = cfg.data( index );
        key.nodeId   = nodeId;
        key.physicalId  = physicalId;
        key.portIndex   = localPortNum;
        key.index       = index;
        gangedLinkEntry.insert(make_pair(key, data));
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseIngressReqTable( const lwSwitch &lwswitch,
                                      uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        for ( i = 0; i < access.reqrte_size(); i++)
        {
            ec = parseOneIngressReqEntry( nodeId, physicalId,
                                     access.has_localportnum() ? access.localportnum() : port,
                                     access.reqrte(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse access port ingress request entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d, local port num %d, index %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwswitch.trunk(port);
        for ( i = 0; i < trunk.reqrte_size(); i++)
        {
            ec = parseOneIngressReqEntry( nodeId, physicalId,
                                     trunk.has_localportnum() ? trunk.localportnum() : port,
                                     trunk.reqrte(i) );

            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse trunk port ingress request entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d, index %d",
                             nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseIngressRespTable( const lwSwitch &lwswitch,
                                       uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);
        for ( i = 0; i < access.rsprte_size(); i++)
        {
            ec = parseOneIngressRespEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.rsprte(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse access port ingress response entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwswitch.trunk(port);
        for ( i = 0; i < trunk.rsprte_size(); i++)
        {
            ec = parseOneIngressRespEntry( nodeId, physicalId,
                                           trunk.has_localportnum() ? trunk.localportnum() : port,
                                           trunk.rsprte(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse trunk port ingress response entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }
    return FM_INT_ST_OK;
}

// new routing tables introduced with LimeRock

FMIntReturn_t
FMFabricParser::parseIngressRmapTable( const lwSwitch &lwSwitch, RemapTable remapTable,
                                       uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FMIntReturn_t ec = FM_INT_ST_OK;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwSwitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);

        switch ( remapTable )
        {
            case NORMAL_RANGE:
            {
                for ( i = 0; i < access.rmappolicytable_size(); i++)
                {
                    ec = parseOneIngressRmapEntry( nodeId, physicalId,
                                                   access.has_localportnum() ? access.localportnum() : port,
                                                   remapTable, access.rmappolicytable(i) );
                    if ( ec != FM_INT_ST_OK )
                    {
                        break;
                    }
                }
                break;
            }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
            case EXTENDED_RANGE_A:
            {
                for ( i = 0; i < access.extarmappolicytable_size(); i++)
                {
                    ec = parseOneIngressRmapEntry( nodeId, physicalId,
                                                   access.has_localportnum() ? access.localportnum() : port,
                                                   remapTable, access.extarmappolicytable(i) );
                    if ( ec != FM_INT_ST_OK )
                    {
                        break;
                    }
                }
                break;
            }

            case EXTENDED_RANGE_B:
            {
                for ( i = 0; i < access.extbrmappolicytable_size(); i++)
                {
                    ec = parseOneIngressRmapEntry( nodeId, physicalId,
                                                   access.has_localportnum() ? access.localportnum() : port,
                                                   remapTable, access.extbrmappolicytable(i) );
                    if ( ec != FM_INT_ST_OK )
                    {
                        break;
                    }
                }
                break;
            }
#endif
            default:
                break;
            }

            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse ingress route map entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseIngressRidTable( const lwSwitch &lwSwitch,
                                       uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwSwitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);
        for ( i = 0; i < access.ridroutetable_size(); i++)
        {
            ec = parseOneIngressRidEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.ridroutetable(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse access port ingress route id entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }

    for ( port = 0; port < lwSwitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwSwitch.trunk(port);
        for ( i = 0; i < trunk.ridroutetable_size(); i++)
        {
            ec = parseOneIngressRidEntry( nodeId, physicalId,
                                           trunk.has_localportnum() ? trunk.localportnum() : port,
                                           trunk.ridroutetable(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse trunk port ingress route id entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseIngressRlanTable( const lwSwitch &lwSwitch,
                                         uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwSwitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);
        for ( i = 0; i < access.rlanroutetable_size(); i++)
        {
            ec = parseOneIngressRlanEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.rlanroutetable(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse access port ingress route lan table entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }

    for ( port = 0; port < lwSwitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwSwitch.trunk(port);
        for ( i = 0; i < trunk.rlanroutetable_size(); i++)
        {
            ec = parseOneIngressRlanEntry( nodeId, physicalId,
                                           trunk.has_localportnum() ? trunk.localportnum() : port,
                                           trunk.rlanroutetable(i) );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse trunk port ingress route lan table entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d index %d",
                             nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }
    return FM_INT_ST_OK;
}


FMIntReturn_t
FMFabricParser::parseGangedLinkTable( const lwSwitch &lwswitch,
                                      uint32_t nodeId, uint32_t physicalId )
{
    int port;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);
        if ( access.has_gangedlinktbl() )
        {
            ec = parseOneGangedLinkEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.gangedlinktbl() );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse access port ganged link entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d",
                             nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }

    for ( port = 0; port < lwswitch.trunk_size(); port++ )
    {
        const trunkPort &trunk = lwswitch.trunk(port);
        if ( trunk.has_gangedlinktbl() )
        {
            ec = parseOneGangedLinkEntry( nodeId, physicalId,
                                          trunk.has_localportnum() ? trunk.localportnum() : port,
                                          trunk.gangedlinktbl() );
            if ( ec != FM_INT_ST_OK )
            {
                FM_LOG_ERROR("failed to parse trunk port ganged link entry for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d",
                             nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port);
                return FM_INT_ST_ILWALID_PORT;
            }
        }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOnePort(const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId,
                             int portIndex, int isAccess)
{
    PortKeyType key;
    int localPortNum;
    lwswitch::switchPortInfo *info;
    switchPortConfig *cfg;
    PortType portType;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d lwswitch physicalId %d portIndex %d isAccess %d.",
                 nodeId, physicalId, portIndex, isAccess);

    if ( !IS_PORT_VALID(portIndex) )
    {
        FM_LOG_ERROR("failed to parse route for port, invalid port index " NODE_ID_LOG_STR " %d LWSwitch physical id %d port index %d",
                     nodeId, physicalId, portIndex);
        return FM_INT_ST_ILWALID_PORT;
    }

    localPortNum = portIndex;
    if ( isAccess )
    {
        localPortNum = lwswitch.access(portIndex).has_localportnum() ?
                           lwswitch.access(portIndex).localportnum() : localPortNum;
        portType = ACCESS_PORT_GPU;
    }
    else
    {
        localPortNum = lwswitch.trunk(portIndex).has_localportnum() ?
                           lwswitch.trunk(portIndex).localportnum() : localPortNum;
        portType = TRUNK_PORT_SWITCH;
    }

    if ( !IS_PORT_VALID(localPortNum) )
    {
        FM_LOG_ERROR("failed to parse route for port, invalid local port number for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d",
                     nodeId, physicalId, localPortNum);
        return FM_INT_ST_ILWALID_PORT;
    }

    info = new lwswitch::switchPortInfo;
    cfg  = new switchPortConfig;

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;

    portInfo.insert(make_pair(key, info));
    info->set_port( localPortNum );

    if ( isAccess && lwswitch.access(portIndex).has_config() )
    {
        cfg->CopyFrom( lwswitch.access(portIndex).config() );
    }
    else if ( !isAccess && lwswitch.trunk(portIndex).has_config() )
    {
        cfg->CopyFrom( lwswitch.trunk(portIndex).config() );
    }
    else
    {
        FM_LOG_ERROR("failed to parse route for port, invalid port config for " NODE_ID_LOG_STR " %d LWSwitch physical id %d local port num %d",
                     nodeId, physicalId, localPortNum);
        delete cfg;
        return FM_INT_ST_ILWALID_PORT_CFG;
    }

    info->set_allocated_config( cfg );
    return FM_INT_ST_OK;
}

// construct ports to reach a GPU from the access port info
void
FMFabricParser::constructPortsToReachGpuMap(uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum, const accessPort &access)
{
    if ( !access.has_farpeerid() )
    {
        return;

    }
    uint32_t targetId = access.farpeerid();

    // construct GPU reachablity map from the access port
    SwitchKeyType switchKey;
    switchKey.nodeId  = nodeId;
    switchKey.physicalId = physicalId;

    PortsToReachGpuMap::iterator it = portsToReachGpuMap.find(switchKey);
    if ( it == portsToReachGpuMap.end() )
    {
        PortsToReachGpu portsToReachGpu;
        portsToReachGpu.clear();
        portsToReachGpuMap.insert( make_pair(switchKey, portsToReachGpu) );
        it = portsToReachGpuMap.find(switchKey);
    }

    PortsToReachGpu &portsToReachGpu = it->second;
    PortsToReachGpu::iterator jit = portsToReachGpu.find( targetId );
    if ( jit == portsToReachGpu.end() )
    {
        portsToReachGpu.insert( make_pair(targetId, 0) );
        jit = portsToReachGpu.find( targetId );
    }
    uint64_t portMask = jit->second;
    portMask |= (uint64_t)1 << localPortNum;
    portsToReachGpu[targetId] = portMask;
}

// construct GPUs that are reachable from a port from the access port info
void
FMFabricParser::consructGpusReachableFromPortMap(uint32_t nodeId, uint32_t physicalId,
                                                 uint32_t localPortNum, const accessPort &access)
{
    if ( !access.has_farpeerid() )
    {
        return;

    }
    uint32_t targetId = access.farpeerid();

    // construct GPU reachablity map from the access port
    SwitchKeyType switchKey;
    switchKey.nodeId  = nodeId;
    switchKey.physicalId = physicalId;

    GpusReachableFromPortMap::iterator it = gpusReachableFromPortMap.find(switchKey);
    if ( it == gpusReachableFromPortMap.end() )
    {
        GpusReachableFromPort gpusReachableFromPort;
        gpusReachableFromPort.clear();
        gpusReachableFromPortMap.insert( make_pair(switchKey, gpusReachableFromPort) );
        it = gpusReachableFromPortMap.find(switchKey);
    }

    GpusReachableFromPort &gpusReachableFromPort = it->second;
    GpusReachableFromPort::iterator jit = gpusReachableFromPort.find( localPortNum );
    if ( jit == gpusReachableFromPort.end() )
    {
        GpuTargetIds gpus;
        gpus.clear();
        gpusReachableFromPort.insert( make_pair(localPortNum, gpus) );
        jit = gpusReachableFromPort.find( localPortNum );
    }

    GpuTargetIds &gpus = jit->second;
    gpus.insert( targetId );
}

FMIntReturn_t
FMFabricParser::parseAccessPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    int i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    for ( i = 0; i < lwswitch.access_size(); i++ )
    {
        ec = parseOnePort(lwswitch, nodeId, physicalId, i, 1);
        if ( ec != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to parse access port for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d with error %d",
                         nodeId, physicalId, i, ec);
            return ec;
        }

        // use the access port info to construct GPU reachability map
        const accessPort &access = lwswitch.access(i);
        constructPortsToReachGpuMap(nodeId, physicalId, access.localportnum(), access);
        consructGpusReachableFromPortMap(nodeId, physicalId, access.localportnum(), access);
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseTrunkPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    int i;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    for ( i = 0; i < lwswitch.trunk_size(); i++ )
    {
        ec = parseOnePort(lwswitch, nodeId, physicalId, i, 0);
        if ( ec != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to parse trunk port for " NODE_ID_LOG_STR " %d LWSwitch physical id %d port %d with error %d",
                         nodeId, physicalId, i, ec);
            return ec;
        }
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneLwswitch( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    lwswitch::switchInfo *pInfo = new lwswitch::switchInfo;
    switchConfig *pConfig;
    int i;
    SwitchKeyType key;
    FMIntReturn_t ec;

    FM_LOG_DEBUG(NODE_ID_LOG_STR " %d, lwswitch physicalId %d ",
                 nodeId, physicalId);

    key.nodeId  = nodeId;
    key.physicalId = physicalId;

    pInfo->set_switchphysicalid( physicalId );
    lwswitchCfg.insert(make_pair(key, pInfo));

    // global switch config
    if ( lwswitch.has_config() )
    {
        pConfig = new switchConfig;
        pConfig->CopyFrom( lwswitch.config() );
        pInfo->set_allocated_config( pConfig );
    }

    // access ports
    ec = parseAccessPorts( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    // trunk ports
    ec = parseTrunkPorts( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    // Ingress request table
    ec = parseIngressReqTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    // Ingress response table
    ec = parseIngressRespTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;

    }

    // Ingress remap/policy table
    ec = parseIngressRmapTable( lwswitch, NORMAL_RANGE, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    ec = parseIngressRmapTable( lwswitch, EXTENDED_RANGE_A, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    ec = parseIngressRmapTable( lwswitch, EXTENDED_RANGE_B, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }
#endif

    // Ingress route ID table
    ec = parseIngressRidTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    // Ingress route link aggregate number table
    ec = parseIngressRlanTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    // Ganged link table
    ec = parseGangedLinkTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_INT_ST_OK )
    {
        // error already logged
        return ec;
    }

    return ec;
}

int
FMFabricParser::checkLWLinkConnExists( TopologyLWLinkConnList &connList,
                                       TopologyLWLinkConn newConn )
{
    TopologyLWLinkConnList::iterator it;

    for ( it = connList.begin(); it != connList.end(); it++ )
    {
        TopologyLWLinkConn tempConn = *it;
        if ((tempConn.localEnd== newConn.localEnd) &&
            (tempConn.farEnd == newConn.farEnd))
        {
            return 1;
        }
        // do reverse comparison as well
        if ((tempConn.localEnd == newConn.farEnd) &&
            (tempConn.farEnd == newConn.localEnd))
        {
            return 1;
        }
    }

    return 0;
}

void
FMFabricParser::copyAccessPortConnInfo( accessPort &accessPort, uint32_t nodeId,
                                        uint32_t physicalId,int portIndex,
                                        TopologyLWLinkConnList &connList )
{
    TopologyLWLinkConn connInfo;
    int localPortNum;

    localPortNum = accessPort.has_localportnum() ?accessPort.localportnum():portIndex;
    connInfo.localEnd.nodeId = nodeId;
    connInfo.localEnd.lwswitchOrGpuId = physicalId;
    connInfo.localEnd.portIndex = localPortNum;

    connInfo.farEnd.nodeId = accessPort.farnodeid();
    connInfo.farEnd.lwswitchOrGpuId = accessPort.farpeerid();
    connInfo.farEnd.portIndex = accessPort.farportnum();
    connInfo.connType = ACCESS_PORT_GPU;
    connList.push_back(connInfo);
}

void
FMFabricParser::copyTrunkPortConnInfo( trunkPort &trunkPort, uint32_t nodeId,
                                       uint32_t physicalId,int portIndex,
                                       TopologyLWLinkConnList &connList )
{
    TopologyLWLinkConn connInfo;
    int localPortNum;

    localPortNum = trunkPort.has_localportnum() ?trunkPort.localportnum():portIndex;
    connInfo.localEnd.nodeId = nodeId;
    connInfo.localEnd.lwswitchOrGpuId = physicalId;
    connInfo.localEnd.portIndex = localPortNum;

    connInfo.farEnd.nodeId = trunkPort.farnodeid();
    connInfo.farEnd.lwswitchOrGpuId = trunkPort.farswitchid();
    connInfo.farEnd.portIndex = trunkPort.farportnum();
    connInfo.connType = TRUNK_PORT_SWITCH;
    // eliminate duplicate trunk connections, ie lwswitch-to-lwswitch connection
    // in the same node which appears in topology
    if (!checkLWLinkConnExists(connList, connInfo))
    {
        connList.push_back(connInfo);
    }

    // found a trunk connection, compute/update our LWSwitch trunk port information
    updateSwitchTrunkPortMaskInfo(connInfo);
}

void
FMFabricParser::updateSwitchTrunkPortMaskInfo(TopologyLWLinkConn &connInfo)
{
    TopologySwitchTrunkPortMaskInfo::iterator it;
    SwitchKeyType switchKey;

    // only interested in trunk connection
    if (connInfo.connType != TRUNK_PORT_SWITCH) {
        return;
    }

    // first update localend switch information
    switchKey.nodeId = connInfo.localEnd.nodeId;
    switchKey.physicalId = connInfo.localEnd.lwswitchOrGpuId;
    it = mSwitchTrunkPortMaskInfo.find(switchKey);
    if (it != mSwitchTrunkPortMaskInfo.end()) {
        // found the switch, update/append to the existing mask
        uint64_t trunkPortMask = it->second;
        trunkPortMask |= BIT64(connInfo.localEnd.portIndex);
        // remove and reinsert with updated mask information
        mSwitchTrunkPortMaskInfo.erase(it);
        mSwitchTrunkPortMaskInfo.insert(std::make_pair(switchKey, trunkPortMask));
    } else {
        // first connection for this LWSwitch, add to the map
        uint64_t trunkPortMask = 0;
        trunkPortMask |= BIT64(connInfo.localEnd.portIndex);
        mSwitchTrunkPortMaskInfo.insert(std::make_pair(switchKey, trunkPortMask));
    }

    // update farend switch information
    switchKey.nodeId = connInfo.farEnd.nodeId;
    switchKey.physicalId = connInfo.farEnd.lwswitchOrGpuId;
    it = mSwitchTrunkPortMaskInfo.find(switchKey);
    if (it != mSwitchTrunkPortMaskInfo.end()) {
        // found the switch, update/append to the existing mask
        uint64_t trunkPortMask = it->second;
        trunkPortMask |= BIT64(connInfo.farEnd.portIndex);
        // remove and re insert with updated mask information
        mSwitchTrunkPortMaskInfo.erase(it);
        mSwitchTrunkPortMaskInfo.insert(std::make_pair(switchKey, trunkPortMask));
    } else {
        // first connection for this LWSwitch, add to the map
        uint64_t trunkPortMask = 0;
        trunkPortMask |= BIT64(connInfo.farEnd.portIndex);
        mSwitchTrunkPortMaskInfo.insert(std::make_pair(switchKey, trunkPortMask));
    }
}

FMIntReturn_t
FMFabricParser::getSwitchTrunkLinkMask(uint32_t nodeId, uint32_t physicalId, uint64 &trunkLinkMask)
{
    TopologySwitchTrunkPortMaskInfo::iterator it;
    SwitchKeyType switchKey;

    switchKey.nodeId = nodeId;
    switchKey.physicalId = physicalId;
    it = mSwitchTrunkPortMaskInfo.find(switchKey);
    if (it != mSwitchTrunkPortMaskInfo.end()) {
        // found the switch
        trunkLinkMask = it->second;
        return FM_INT_ST_OK;
    }

    // not found the specified LWSwitch
    trunkLinkMask = 0;
    return FM_INT_ST_ILWALID_LWSWITCH;
}

FMIntReturn_t
FMFabricParser::parseLWLinkConnections( const node &node, uint32_t nodeId )
{
    TopologyLWLinkConnList lwlinkConnList;

    // create the connection list by going through all the access and trunk
    // port of each lwswitch
    for ( int i = 0; i < node.lwswitch_size(); i++ )
    {
        lwSwitch lwswitch = node.lwswitch(i);
        // use index as physicalId if it is not present
        uint32_t physicalId = i;
        if ( lwswitch.has_physicalid() )
        {
            physicalId = lwswitch.physicalid();
        }

        // parse access connections
        for ( int j = 0; j < lwswitch.access_size(); j++ )
        {
            copyAccessPortConnInfo((accessPort&)lwswitch.access(j), nodeId, physicalId, j, lwlinkConnList);
        }
        // parse trunk connections
        for ( int k = 0; k < lwswitch.trunk_size(); k++ )
        {
            copyTrunkPortConnInfo((trunkPort&)lwswitch.trunk(k), nodeId, physicalId, k, lwlinkConnList);
        }
    }

    lwLinkConnMap.insert( std::make_pair(nodeId, lwlinkConnList) );
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseSharedLwswitchPartitions(uint32_t nodeId, nodeSystemPartitionInfo &partitionInfo)
{
    // parse the configured partition or customer provided partitions
    for ( int i = 0; i < partitionInfo.sharedlwswitchinfo_size(); i++ )
    {
        PartitionKeyType key;
        key.nodeId = nodeId;

        const sharedLWSwitchPartitionInfo &partCfgInfo = partitionInfo.sharedlwswitchinfo(i);
        key.partitionId = partCfgInfo.has_partitionid() ? partCfgInfo.partitionid() : i;

        sharedLWSwitchPartitionInfo *partInfo = new sharedLWSwitchPartitionInfo;
        partInfo->CopyFrom( partCfgInfo );

        sharedLwswitchPartitionCfg.insert(make_pair(key, partInfo));
    }
    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseOneNode( const node &node, int nodeIndex )
{
    int i;
    FMIntReturn_t ec;
    NodeKeyType key;
    NodeConfig *pNode;
    uint32_t nodeId = node.has_nodeid() ? node.nodeid() : nodeIndex;

    FM_LOG_DEBUG("fabricNode index %d, nodeId %d.", nodeIndex, nodeId);

    pNode = new NodeConfig;
    pNode->nodeId = nodeId;
    key.nodeId = nodeId;

    if (node.has_ipaddress())
    {
        pNode->IPAddress = new std::string( node.ipaddress() );
    }
    else {
        pNode->IPAddress = new std::string( FM_DEFAULT_BIND_INTERFACE );
    }

    NodeCfg.insert(make_pair(key, pNode));

    pNode->partitionInfo.Clear();
    if ( node.has_partitioninfo() )
    {
        // use the one from the topology file
        pNode->partitionInfo = node.partitioninfo();
    }

    if ( mpPartitionInfo )
    {
        int i;

        // remove the shared partitions from the topology
        google::protobuf::RepeatedPtrField<sharedLWSwitchPartitionInfo> *partInfoes = pNode->partitionInfo.mutable_sharedlwswitchinfo();
        int sharedPartitionSize = pNode->partitionInfo.sharedlwswitchinfo_size();
        for ( i = 0; i < sharedPartitionSize; i++ )
        {
            partInfoes->RemoveLast();
        }

        // overwrite with the customer shared partition info
        for ( i = 0; i < mpPartitionInfo->sharedlwswitchinfo_size(); i++ )
        {
            sharedLWSwitchPartitionInfo *pInfo = pNode->partitionInfo.add_sharedlwswitchinfo();
            pInfo->CopyFrom( mpPartitionInfo->sharedlwswitchinfo(i) );
        }
    }

    // parse shared LWSwitch partitions
    parseSharedLwswitchPartitions(nodeId, pNode->partitionInfo);


    //validate the total number of GPUs, and LWSwitches present in the topology file
    if ( node.gpu_size() > MAX_NUM_GPUS_PER_NODE )
    {
        FM_LOG_ERROR("failed to parse fabric topology as number of GPUs to configure is more than allowed");
        return FM_INT_ST_ILWALID_GPU;
    }

    if ( node.lwswitch_size() > MAX_NUM_LWSWITCH_PER_NODE )
    {
        FM_LOG_ERROR("failed to parse fabric topology as number of LWSwitches to configure is more than allowed");
        return FM_INT_ST_ILWALID_LWSWITCH;
    }

    // configure all the GPUs
    for ( i = 0; i < node.gpu_size(); i++ )
    {
        ec = parseOneGpu( node.gpu(i), nodeId, i );
        if ( ec != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to parse GPU config from fabric topology for " NODE_ID_LOG_STR " %d GPU index %d error %d",
                         nodeId, i, ec);
            return ec;
        }
    }

    // configure all the LWSwitches
    for ( i = 0; i < node.lwswitch_size(); i++ )
    {
        lwSwitch lwswitch = node.lwswitch(i);
        // use index as physicalId if it is not present
        uint32_t physicalId = i;
        if ( lwswitch.has_physicalid() )
        {
            physicalId = lwswitch.physicalid();
#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
            if (lwswitch.has_slotid()) {
                mSwitchPhyiscalIdToSlotIdMapping[physicalId] = lwswitch.slotid();
            }
#endif
        }

        ec = parseOneLwswitch(lwswitch , nodeId, physicalId );
        if ( ec != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to parse LWSwitch config from fabric topology for " NODE_ID_LOG_STR " %d LWSwitch physical id %d error %d",
                         nodeId, i, ec);
            return ec;
        }
    }

    ec = parseLWLinkConnections(node, nodeId);
    if ( ec != FM_INT_ST_OK )
    {
        FM_LOG_ERROR("failed to parse LWLink connections from fabric topology for " NODE_ID_LOG_STR " %d error %d",
                     nodeId, ec);
        return ec;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::validateLwstomSharedFabricPartition( void )
{
    if ( mpPartitionInfo->has_version() &&
         ( mpPartitionInfo->version() != FABRIC_PARTITION_VERSION ) )
    {
        FM_LOG_ERROR("version %d specified in custom partition definition file %s does not match.",
                     mpPartitionInfo->version(), mFabricPartitionFileName);
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    int numPartitions = mpPartitionInfo->sharedlwswitchinfo_size();
    if ( ( numPartitions == 0 ) || ( numPartitions > FM_MAX_FABRIC_PARTITIONS ) )
    {
        FM_LOG_ERROR("total number of shared fabric partitions %d specified in custom partition definition file %s is invalid.",
                     numPartitions, mFabricPartitionFileName);
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    int numMaxGpuPartitions = 0;
    int num1GpuPartitions = 0;

    for ( int i = 0; i < numPartitions; i++ )
    {
        sharedLWSwitchPartitionInfo partInfo = mpPartitionInfo->sharedlwswitchinfo(i);

        if ( partInfo.has_partitionid() == false )
        {
            FM_LOG_ERROR("missing partition id information in custom partition definition file %s.",
                         mFabricPartitionFileName);
            return FM_INT_ST_FILE_PARSING_ERR;
        }

        int numGpus = partInfo.gpuinfo_size();
        if ( ( numGpus != 1 ) && ( numGpus != 2 ) && ( numGpus != 4 ) &&
             ( numGpus != 8 ) && ( numGpus != 16 ) )
        {
            FM_LOG_ERROR("number of GPU count %d specified in partition id %d is not supported in custom partition definition file %s.",
                         numGpus, partInfo.partitionid(), mFabricPartitionFileName);
            return FM_INT_ST_FILE_PARSING_ERR;
        }

        if ( numGpus == MAX_NUM_GPUS_PER_NODE ) numMaxGpuPartitions++;
        if ( numGpus == 1 ) num1GpuPartitions++;
    }

    if ( numMaxGpuPartitions > 1 )
    {
        FM_LOG_ERROR("total number of %d GPU partitions is more than the supported limit in custom partition definition file %s.",
                     MAX_NUM_GPUS_PER_NODE, mFabricPartitionFileName);
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    if ( num1GpuPartitions > MAX_NUM_GPUS_PER_NODE )
    {
        FM_LOG_ERROR("total number of 1 GPU partitions is more than the supported limit in custom partition definition file %s.",
                     mFabricPartitionFileName);
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseLwstomSharedFabricPartition( void )
{
    if ( mFabricPartitionFileName == NULL )
    {
        // overwrite partition is not specified
        return FM_INT_ST_OK;
    }

    mpPartitionInfo = new( nodeSystemPartitionInfo );

    // Read the protobuf binary file.
    fstream input(mFabricPartitionFileName, ios::in | ios::binary);
    if ( !input )
    {
        FM_LOG_ERROR("failed to open custom shared fabric partition file %s", mFabricPartitionFileName);
        FM_SYSLOG_ERR("failed to open custom shared fabric partition file %s", mFabricPartitionFileName);
        return FM_INT_ST_FILE_OPEN_ERR;
    }
    else if ( !mpPartitionInfo->ParseFromIstream(&input) )
    {
        FM_LOG_ERROR("failed to parse custom shared fabric partition file %s", mFabricPartitionFileName);
        FM_SYSLOG_ERR("failed to parse custom shared fabric partition file %s", mFabricPartitionFileName);
        input.close();
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    input.close();
    FM_LOG_INFO("parsed custom shared fabric partition file %s successfully, name %s, version %d, time %s",
                mFabricPartitionFileName,
                mpPartitionInfo->has_name() ? mpPartitionInfo->name().c_str() : "Not set",
                mpPartitionInfo->has_version() ? mpPartitionInfo->version() : -1,
                mpPartitionInfo->has_time() ? mpPartitionInfo->time().c_str() : "Not set");

    return validateLwstomSharedFabricPartition();
}

FMIntReturn_t
FMFabricParser::parseFabricTopology( const char *topoFile )
{
    int i;
    FMIntReturn_t ec;

    if ( !topoFile )
    {
        FM_LOG_ERROR("fabric topology file name/path is null");
        return FM_INT_ST_FILE_ILWALID;
    }

    // Read the protobuf binary file.
    fstream input(topoFile, ios::in | ios::binary);
    if ( !input )
    {
#if defined(LW_MODS) && !defined(LW_MODS_GDM_BUILD)
        vector<char> data;
        ec = ReadTarFile(topoFile, &data);
        if ( ec == FM_INT_ST_OK )
        {
            IMemStream memStream(&data[0], data.size());
            if ( !mpFabric->ParseFromIstream(&memStream) )
            {
                FM_LOG_ERROR("failed to parse fabric topology file %s", topoFile);
                FM_SYSLOG_ERR("failed to parse fabric topology file %s", topoFile);
                return FM_INT_ST_FILE_PARSING_ERR;
            }
        }
        else
#endif
        {
            FM_LOG_ERROR("failed to open fabric topology file %s", topoFile);
            FM_SYSLOG_ERR("failed to open fabric topology file %s", topoFile);
            return FM_INT_ST_FILE_OPEN_ERR;
        }
    }
    else if ( !mpFabric->ParseFromIstream(&input) )
    {
        FM_LOG_ERROR("failed to parse fabric topology file %s", topoFile);
        FM_SYSLOG_ERR("failed to parse fabric topology file %s", topoFile);
        input.close();
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    input.close();
    FM_LOG_INFO("parsed fabric topology file %s successfully. topology name: %s, build time: %s.",
                topoFile,
                mpFabric->has_name() ? mpFabric->name().c_str() : "Not set",
                mpFabric->has_time() ? mpFabric->time().c_str() : "Not set");

    // parse the customer defined shared fabric partition table if it is specified
    ec = parseLwstomSharedFabricPartition();
    if ( ec != FM_INT_ST_OK )
    {
        // errors are already logged
        return FM_INT_ST_FILE_PARSING_ERR;
    }

    for (int i = 0; i < mpFabric->fabricnode_size(); i++)
    {
        ec = parseOneNode ( mpFabric->fabricnode(i), i );
        if ( ec != FM_INT_ST_OK )
        {
            FM_LOG_ERROR("failed to parse " NODE_ID_LOG_STR " %d from fabric topology file with error %d", i, ec);
            return ec;
        }
    }

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
    // store osfpPort mapping info
    getLwlinkToOsfpPortMappingInfo();
#endif

    // this copy is created because we need some information about a switch
    // even if it is excluded, which would result in it getting pruned from the original
    mpFabricCopy = mpFabric->New();
    mpFabricCopy->CopyFrom(*mpFabric);

    return FM_INT_ST_OK;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
void
FMFabricParser::getLwlinkToOsfpPortMappingInfo()
{
    int i;

    for (i = 0; i < mpFabric->portmap_size(); i++) {
        lwlinkToOsfpPortMapping portMap = mpFabric->portmap(i);
        FM_LOG_DEBUG("linkindex %d port number %d", portMap.linkindex(), portMap.externalportnum());
        mOsfpPortMap[portMap.linkindex()] = portMap.externalportnum();
    }

    return;
}
uint32_t
FMFabricParser::getOsfpPortNumForLinkIndex(uint32_t linkIndex)
{
    uint32_t osfpPortNumber = -1;

    if (mOsfpPortMap.find(linkIndex) != mOsfpPortMap.end()) {
        osfpPortNumber = mOsfpPortMap[linkIndex];
        FM_LOG_DEBUG("linkindex %d port number %d", linkIndex, osfpPortNumber);
    }

    return osfpPortNumber;
}

bool
FMFabricParser::getSlotId(uint32_t physicalId, uint32_t &slotId)
{
    bool ret = false;
    std::map <uint32_t, uint32_t>::iterator it;
    it = mSwitchPhyiscalIdToSlotIdMapping.find(physicalId);

    if (it != mSwitchPhyiscalIdToSlotIdMapping.end()) {
        ret = true;
        slotId = it->second;
    }

    return ret;
}

bool
FMFabricParser::isSlotIdProvided()
{
    return mSwitchPhyiscalIdToSlotIdMapping.size() > 0;
}

void
FMFabricParser::updateFabricNodeAddressInfo(std::map<uint32_t, string> nodeToIpAddrMap, std::set<uint32_t> &degradedNodes)
{
    std::map <NodeKeyType,      NodeConfig *>::iterator it;
    for (it = NodeCfg.begin(); it != NodeCfg.end();) {
        NodeKeyType key = it->first;

        if (nodeToIpAddrMap.find(key.nodeId) == nodeToIpAddrMap.end()) {
            // nodeId mentioned in the topology file is not specified in fabric address file
            // degrade that node by removing it from topology
            degradedNodes.insert(key.nodeId);
            it = NodeCfg.erase(it);
            FM_LOG_INFO("degrading " NODE_ID_LOG_STR " %d as it is not found in fabric node config file", key.nodeId);
            continue;
        }


        NodeConfig *pNode = it->second;
        // delete already allocated memory for the IP address
        delete(pNode->IPAddress);
        pNode->IPAddress = new std::string(nodeToIpAddrMap[key.nodeId]);
        ++it;
    }
}
#endif

#ifdef DEBUG
FMIntReturn_t
FMFabricParser::parseDisableGPUConf( std::vector<std::string> &gpus, lwSwitchArchType arch )
{
    int i, j, rc, nodeId, physicalId;
    GpuKeyType key;

    // i, starting from 1, as 0 is str --disable-gpu
    for ( i = 1; i < (int)gpus.size(); i++ )
    {
        rc = sscanf( gpus[i].c_str(), "%d/%d", &nodeId, &physicalId );
        if ( ( rc != 2 ) || !IS_GPU_VALID( physicalId ) )
        {
            FM_LOG_DEBUG("parseDisableGPUConf - Invalid GPU " NODE_ID_LOG_STR " %d physical id %d", nodeId, physicalId);
            continue;
        }

        key.nodeId = nodeId;
        key.physicalId = physicalId;

        disableGpu( key, arch );
        FM_LOG_DEBUG("parseDisableGPUConf - GPU " NODE_ID_LOG_STR " %d physical id %d is disabled.", nodeId, physicalId);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseDisableSwitchConf( std::vector<std::string> &switches, lwSwitchArchType arch )
{
    int i, j, rc, nodeId, physicalId;
    SwitchKeyType key;

    // i, starting from 1, as 0 is str --disable-switch
    for ( i = 1; i < (int)switches.size(); i++ )
    {
        rc = sscanf( switches[i].c_str(), "%d/%x", &nodeId, &physicalId );
        if ( ( rc != 2 ) )
        {
            FM_LOG_DEBUG("parseDisableSwitchConf - Invalid switch " NODE_ID_LOG_STR " %d physical id %d", nodeId, physicalId);
            continue;
        }

        key.nodeId = nodeId;
        key.physicalId = physicalId;
        disableSwitch( key );
        FM_LOG_DEBUG("parseDisableSwitchConf - Switch " NODE_ID_LOG_STR " %d physical id %d is disabled.", nodeId, physicalId);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseDisablePartitionConf( std::vector<std::string> &partitions )
{
    int i, j, rc, nodeId, partitionId;
    PartitionKeyType key;

    // i, starting from 1, as 0 is str --disable-partition
    for ( i = 1; i < (int)partitions.size(); i++ )
    {
        rc = sscanf( partitions[i].c_str(), "%d/%d", &nodeId, &partitionId );
        if ( ( rc != 2 ) )
        {
            FM_LOG_DEBUG("parseDisablePartitionConf - Invalid partition with " NODE_ID_LOG_STR " %d partition id %d", nodeId, partitionId);
            continue;
        }

        key.nodeId = nodeId;
        key.partitionId = partitionId;
        disablePartition( key );
        FM_LOG_DEBUG("parseDisablePartitionConf - Partition with " NODE_ID_LOG_STR " %d partition id %d is disabled.", nodeId, partitionId);
    }

    return FM_INT_ST_OK;
}

FMIntReturn_t
FMFabricParser::parseLoopbackPorts( std::vector<std::string> &ports )
{
    int i, rc, nodeId, physicalId, portIndex;
    PortKeyType key;

    // i, starting from 1, as 0 is str --loopback-port
    for ( i = 1; i < (int)ports.size(); i++ )
    {
        rc = sscanf( ports[i].c_str(), "%d/%d/%d", &nodeId, &physicalId, &portIndex);
        if ( ( rc != 3 ) || !IS_PORT_VALID( portIndex ) )
        {
            FM_LOG_DEBUG("parseLoopbackPorts - Invalid port with " NODE_ID_LOG_STR " %d partition id %d and port index %d", nodeId, physicalId, portIndex);
            continue;
        }

        key.nodeId  = nodeId;
        key.physicalId = physicalId;
        key.portIndex  = portIndex;

        mLoopbackPorts.insert(key);
        FM_LOG_DEBUG("parseLoopbackPorts - " NODE_ID_LOG_STR " %d partition id %d and port index %d is put to loopback.", nodeId, physicalId, portIndex);
    }

    return FM_INT_ST_OK;
}

/*
 * Parse any modification to the topology file already loaded
 * The modification is in a text file topoConfFile, in the following format
 *
 * To prune a list of GPUs, where x is nodeId,
 *                                y is the GPU physicalId on the node
 * --disable-gpu x/y
 *
 *
 * To make a list of port loop back, x is nodeId,
 *                                   y is the switch physicalId on the node
 *                                   z is the localport number on the switch
 * --loopback-port x/y/z
 */


FMIntReturn_t
FMFabricParser::parseFabricTopologyConf( const char *topoConfFileName, lwSwitchArchType arch )
{
    uint i;
    FMIntReturn_t ec;

    if ( !topoConfFileName )
    {
        FM_LOG_ERROR("Invalid topology conf file.");
        return FM_INT_ST_FILE_ILWALID;
    }

    // Read the topoConfFile text file.
    ifstream inFile;
    inFile.open(topoConfFileName);

    if ( inFile.is_open() )
    {
        std::string line;

        while ( getline (inFile,line) )
        {
            if ( line.find(OPT_DISABLE_GPU) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> gpus((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisableGPUConf(gpus, arch);
            }
            else if ( line.find(OPT_DISABLE_SWITCH) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> switches((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisableSwitchConf(switches, arch);
            }
            else if ( line.find(OPT_PORT_LOOPBACK) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> ports((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseLoopbackPorts(ports);
            }
            else
            {
                FM_LOG_DEBUG("parseFabricTopologyConf - Unknown option %s.", line.c_str());
            }

        }
        inFile.close();
    }

    FM_LOG_DEBUG("Parsed file %s successfully.", topoConfFileName);
    return FM_INT_ST_OK;
}

#endif

/*
 * ilwalidate ingress request entry if the port is connected to a pruned GPU
 * make outgoing port to be the port itself, if the port is in loopback
 */
void
FMFabricParser::modifyOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                          uint32_t localPortNum, ingressRequestTable *entry )
{
    PortKeyType portKey;
    int index = entry->index();

    if ( mIlwalidIngressReqEntries.find(index) != mIlwalidIngressReqEntries.end() )
    {
        FM_LOG_DEBUG("ilwalidate " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.",
                     nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress request entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        FM_LOG_DEBUG("loopback " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.\n",
                     nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback, set the outgoing port to be itself
        // Need to clear the old values first.
        entry->set_vcmodevalid7_0(0);
        entry->set_vcmodevalid15_8(0);
        entry->set_vcmodevalid17_16(0);

        if ( localPortNum < 8 )
        {
            entry->set_vcmodevalid7_0(1<< (4*(localPortNum)));
        }
        else if ( localPortNum < 16 )
        {
            entry->set_vcmodevalid15_8(1<< (4*(localPortNum - 8)));
        }
        else
        {
            entry->set_vcmodevalid17_16(1<< (4*(localPortNum - 16)));
        }
    }
}

/*
 * ilwalidate ingress response entry if the port is connected to a pruned GPU
 * make outgoing port to be the port itself, if the port is in loopback
 */
void
FMFabricParser::modifyOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                           uint32_t localPortNum, ingressResponseTable *entry )
{
    PortKeyType portKey;
    int index = entry->index();

    if ( mIlwalidIngressRespEntries.find(index) != mIlwalidIngressRespEntries.end() )
    {
        FM_LOG_DEBUG("ilwalidate " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.",
                     nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress response entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;
    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        FM_LOG_DEBUG("loopback " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.",
                     nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback, set the outgoing port to be itself
        // Need to clear the old values first.
        entry->set_vcmodevalid7_0(0);
        entry->set_vcmodevalid15_8(0);
        entry->set_vcmodevalid17_16(0);

        if ( localPortNum < 8 )
        {
            entry->set_vcmodevalid7_0(1<< (4*(localPortNum)));
        }
        else if ( localPortNum < 16 )
        {
            entry->set_vcmodevalid15_8(1<< (4*(localPortNum - 8)));
        }
        else
        {
            entry->set_vcmodevalid17_16(1<< (4*(localPortNum - 16)));
        }
    }
}

// new routing tables introduced with LimeRock
void
FMFabricParser::modifyOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           RemapTable remapTable, rmapPolicyEntry *entry, lwSwitchArchType arch )
{
    PortKeyType portKey;
    int index = entry->index();

    switch (remapTable)
    {
        case NORMAL_RANGE:
        {
            if ( mIlwalidIngressRmapEntries.find(index) != mIlwalidIngressRmapEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        case EXTENDED_RANGE_A:
        {
            if ( mIlwalidIngressRmapExtAEntries.find(index) != mIlwalidIngressRmapExtAEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
        case EXTENDED_RANGE_B:
        {
            if ( mIlwalidIngressRmapExtBEntries.find(index) != mIlwalidIngressRmapExtBEntries.end() )
            {
                entry->set_entryvalid(0);
            }
            break;
        }
#endif
        default:
            break;
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        FM_LOG_DEBUG("loopback " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.\n",
                     nodeId, physicalId, localPortNum, index);

        accessPort *port = getAccessPortInfo( nodeId, physicalId, localPortNum );
        if ( ( port != NULL ) && port->has_farpeerid() )
        {
            // port is set to be in loopback, set the outgoing target ID to be
            // the GPU this port is connected to
            entry->set_targetid(port->farpeerid());
        }
    }
}

void
FMFabricParser::modifyOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                          ridRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID

    if ( mIlwalidIngressRidEntries.find(index) != mIlwalidIngressRidEntries.end() )
    {
        FM_LOG_DEBUG("ilwalidate " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.",
                     nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RID entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        FM_LOG_DEBUG("loopback " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.\n",
                     nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback, clear out old port list
        // and add just the self index
        entry->clear_portlist();
        list = entry->add_portlist();
        list->set_vcmap(0);
        list->set_portindex(localPortNum);
    }

}

void
FMFabricParser::modifyOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           rlanRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID
    if ( mIlwalidIngressRlanEntries.find(index) != mIlwalidIngressRlanEntries.end() )
    {
        FM_LOG_DEBUG("ilwalidate " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.",
                     nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RLAN entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        FM_LOG_DEBUG("loopback " NODE_ID_LOG_STR " %d, physicalId %d, localPortNum %d, index %d.\n",
                     nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback. no need for vlan in this case
        // Clear out old group list. leave valid alone in case the port is disabled
        // for other reasons.
        entry->clear_grouplist();

    }
}

void
FMFabricParser::modifyRoutingTable( lwSwitchArchType arch )
{
    int i, j, n, w, p;
    int physicalId, nodeId;

    for ( n = 0; n < mpFabric->fabricnode_size(); n++ )
    {
       const node &fnode = mpFabric->fabricnode(n);
       nodeId = fnode.has_nodeid() ? fnode.nodeid() : n;

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           const lwSwitch &lwswitch = fnode.lwswitch(w);
           physicalId = lwswitch.has_physicalid() ? lwswitch.physicalid() : w;

           for ( i = 0; i < lwswitch.access_size(); i++ )
           {
               // modify all routing entries on an access port
               const accessPort &access = lwswitch.access(i);
               for ( j = 0; j < access.reqrte_size(); j++ )
               {
                   modifyOneIngressReqEntry( nodeId, physicalId,
                                             access.localportnum(),
                                             (ingressRequestTable *) &access.reqrte(j));
               }

               for ( j = 0; j < access.rsprte_size(); j++ )
               {
                   modifyOneIngressRespEntry( nodeId, physicalId,
                                              access.localportnum(),
                                              (ingressResponseTable *) &access.rsprte(j));
               }

               for ( j = 0; j < access.rmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( nodeId, physicalId,
                                              access.localportnum(), NORMAL_RANGE,
                                              (rmapPolicyEntry *) &access.rmappolicytable(j),
                                              arch);
               }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
               for ( j = 0; j < access.extarmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( nodeId, physicalId,
                                              access.localportnum(), EXTENDED_RANGE_A,
                                              (rmapPolicyEntry *) &access.extarmappolicytable(j),
                                              arch);
               }

               for ( j = 0; j < access.extbrmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( nodeId, physicalId,
                                              access.localportnum(), EXTENDED_RANGE_B,
                                              (rmapPolicyEntry *) &access.extbrmappolicytable(j),
                                              arch);
               }
#endif

               for ( j = 0; j < access.ridroutetable_size(); j++ )
               {
                   modifyOneIngressRidEntry( nodeId, physicalId,
                                              access.localportnum(),
                                              (ridRouteEntry *) &access.ridroutetable(j));
               }

               for ( j = 0; j < access.rlanroutetable_size(); j++ )
               {
                   modifyOneIngressRlanEntry( nodeId, physicalId,
                                              access.localportnum(),
                                              (rlanRouteEntry *) &access.rlanroutetable(j));
               }
           }

           for ( i = 0; i < lwswitch.trunk_size(); i++ )
           {
               // modify all routing entries on an trunk port
               const trunkPort &trunk = lwswitch.trunk(i);
               for ( j = 0; j < trunk.reqrte_size(); j++ )
               {
                   modifyOneIngressReqEntry( nodeId, physicalId,
                                             trunk.localportnum(),
                                             (ingressRequestTable *) &trunk.reqrte(j));
               }

               for ( j = 0; j < trunk.rsprte_size(); j++ )
               {
                   modifyOneIngressRespEntry( nodeId, physicalId,
                                              trunk.localportnum(),
                                              (ingressResponseTable *) &trunk.rsprte(j));
               }

               for ( j = 0; j < trunk.ridroutetable_size(); j++ )
               {
                   modifyOneIngressRidEntry( nodeId, physicalId,
                                              trunk.localportnum(),
                                              (ridRouteEntry *) &trunk.ridroutetable(j));
               }

               for ( j = 0; j < trunk.rlanroutetable_size(); j++ )
               {
                   modifyOneIngressRlanEntry( nodeId, physicalId,
                                              trunk.localportnum(),
                                              (rlanRouteEntry *) &trunk.rlanroutetable(j));
               }
           }
       }
    }
}

bool accessPortSort( const accessPort &a, const accessPort &b )
{
    return  (a.has_localportnum() && b.has_localportnum() &&
             (a.localportnum() < b.localportnum()));
}

/*
 * prune an access port from the topology
 * used when a GPU connected to an access port is pruned
 */
void
FMFabricParser::pruneOneAccessPort ( uint32_t nodeId, uint32_t physicalId,
                                       lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        FM_LOG_DEBUG("pruneOneAccessPort - Invalid lwswitch.");
        return;
    }

    FM_LOG_DEBUG("removing GPU access port %d/%d/%d from routing configuration",
                 nodeId, physicalId, localportNum);

    for ( i = 0; i < pLwswitch->access_size(); i++ )
    {
        const accessPort &access = pLwswitch->access(i);

        if ( access.has_localportnum() && access.localportnum() == localportNum )
            break;
    }

    if ( i >= pLwswitch->access_size() )
    {
        // access port with the specified localportNum is not found
        return;
    }

    google::protobuf::RepeatedPtrField<accessPort> *ports = pLwswitch->mutable_access();

    if ( i < pLwswitch->access_size() - 1 )
    {
        ports->SwapElements(i, pLwswitch->access_size() - 1);
    }
    ports->RemoveLast();

    // Reorder the list by local port number
    std::sort(ports->begin(), ports->end(), accessPortSort);

    // remove the port from the config
    PortKeyType key;
    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex = localportNum;

    std::map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    it = portInfo.find( key );
    if ( it != portInfo.end() )
    {
        FM_LOG_DEBUG("removed GPU access port %d/%d/%d from routing configuration",
                    key.nodeId, key.physicalId, key.portIndex);
        portInfo.erase( it );
    }
}

/*
 * prune all access ports that are directly connected to a pruned GPU
 */
void
FMFabricParser::pruneAccessPorts( void )
{
    int j, n, w;
    uint32_t i, physicalId, nodeId;
    accessPort *port = NULL;

    if ( mDisabledGpuEndpointIds.size() == 0 )
        return;

    for ( n = 0; n < mpFabric->fabricnode_size(); n++ )
    {
       const node &fnode = mpFabric->fabricnode(n);
       nodeId = fnode.has_nodeid() ? fnode.nodeid() : n;

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           lwSwitch *pLwswitch = (lwSwitch *)&fnode.lwswitch(w);

           physicalId = w;
           if ( pLwswitch->has_physicalid() )
           {
               physicalId = pLwswitch->physicalid();
           }

           for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
           {
               for ( j = 0; j < pLwswitch->access_size(); j++ )
               {
                   port = (accessPort *) &pLwswitch->access(j);
                   if ( port->has_localportnum() && ( port->localportnum() == i ) )
                   {
                       break;
                   }
                   else
                   {
                       port = NULL;
                   }
               }

               if ( !port )
               {
                   continue;
               }

               // find all access ports that are connected to a disabled GPU
               // that is the access port farpeerid is the endPointId of a disabled GPU
               if ( port->has_farpeerid() && port->has_localportnum() &&
                    mDisabledGpuEndpointIds.find(port->farpeerid()) != mDisabledGpuEndpointIds.end() )
               {
                   pruneOneAccessPort( nodeId, physicalId, pLwswitch, port->localportnum() );
               }
           }
       }
    }
}

bool gpuSort( const GPU& a, const GPU& b )
{
    return a.fabricaddrbase() < b.fabricaddrbase();
}

void
FMFabricParser::pruneOneGpu ( uint32_t gpuEndpointID )
{
    int i = 0;
    int nodeId = gpuEndpointID / MAX_NUM_GPUS_PER_NODE;
    uint32_t physicalId = gpuEndpointID % MAX_NUM_GPUS_PER_NODE;

    if ( nodeId >= mpFabric->fabricnode_size() )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " %d while removing a GPU from routing configration", nodeId);
        return;
    }

    FM_LOG_INFO("removing GPU %d/%d from routing configuration", nodeId, physicalId);

    node *pNode = (node *)&mpFabric->fabricnode(getNodeIndex(nodeId));
    for ( i = 0; i < pNode->gpu_size(); i++ )
    {
        const GPU &gpu = pNode->gpu(i);

        if ( gpu.has_physicalid() && ( gpu.physicalid()== physicalId ) )
        {
            break;
        }
    }

    if ( i >= pNode->gpu_size() )
    {
        // GPU with the specified fabricBaseAddr is not found
        return;
    }

    google::protobuf::RepeatedPtrField<GPU> *gpus = pNode->mutable_gpu();

    if ( i < pNode->gpu_size() - 1 )
    {
        gpus->SwapElements(i, pNode->gpu_size() - 1);
    }
    gpus->RemoveLast();

    // Reorder the list by fabric address
    std::sort(gpus->begin(), gpus->end(), gpuSort);

    // remove the GPU from the config
    GpuKeyType key;
    key.nodeId  = nodeId;
    key.physicalId = physicalId;

    std::map <GpuKeyType, lwswitch::gpuInfo * >::iterator it;
    it = gpuCfg.find( key );
    if ( it != gpuCfg.end() )
    {
        FM_LOG_INFO("removed GPU with " NODE_ID_LOG_STR " %d physical id %d from routing configuration", nodeId, physicalId);
        gpuCfg.erase( it );
    }
}

void
FMFabricParser::pruneGpus( void )
{
    std::set<uint32_t>::iterator it;

    if ( mDisabledGpuEndpointIds.size() == 0 )
        return;

    for ( it = mDisabledGpuEndpointIds.begin(); it != mDisabledGpuEndpointIds.end(); it++ )
    {
        pruneOneGpu( *it );
    }
}

void
FMFabricParser::insertIlwalidateIngressRmapEntry(RemapTable remapTable, uint32_t index)
{
    switch ( remapTable )
    {
    case NORMAL_RANGE:
        mIlwalidIngressRmapEntries.insert(index);
        break;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    case EXTENDED_RANGE_A:
        mIlwalidIngressRmapExtAEntries.insert(index);
        break;

    case EXTENDED_RANGE_B:
        mIlwalidIngressRmapExtBEntries.insert(index);
        break;

    case MULTICAST_RANGE:
        mIlwalidIngressRmapMcEntries.insert(index);
        break;
#endif

    default:
        break;
    }
}

/*
 * add a GPU's endPointedId to mDisabledGpuEndpointIds
 */
void
FMFabricParser::disableGpu ( GpuKeyType &gpu, lwSwitchArchType arch )
{
    uint32_t i, gpuTargetId = GPU_TARGET_ID(gpu.nodeId, gpu.physicalId);
    mDisabledGpuEndpointIds.insert(gpuTargetId);

    if ( arch == LWSWITCH_ARCH_TYPE_SV10 )
    {
        uint32_t ingressReqIndex = gpuTargetId << 2;
        uint32_t ingressRespIndex = gpuTargetId * FMDeviceProperty::getNumIngressRespEntriesPerGpu(arch);

        for ( i = 0; i < FMDeviceProperty::getNumIngressReqEntriesPerGpu(arch); i++ )
        {
            // ilwalidate ingress request entries to this GPU
            mIlwalidIngressReqEntries.insert( i + ingressReqIndex );
        }

        for ( i = 0; i < FMDeviceProperty::getNumIngressRespEntriesPerGpu(arch); i++ )
        {
            // ilwalidate ingress response entries to this GPU
            mIlwalidIngressRespEntries.insert( i + ingressRespIndex );
        }
    }
    else
    {
        // Ilwalidate GPA remap entries
        for ( i = 0; i < FMDeviceProperty::getNumGpaRemapEntriesPerGpu(arch); i++ )
        {
            insertIlwalidateIngressRmapEntry(FMDeviceProperty::getGpaRemapTbl(arch),
                                             FMDeviceProperty::getGpaRemapIndexFromTargetId(arch, gpuTargetId) + i);
        }

        // Ilwalidate FLA remap entries
        for ( i = 0; i < FMDeviceProperty::getNumFlaRemapEntriesPerGpu(arch); i++ )
        {
            insertIlwalidateIngressRmapEntry(FMDeviceProperty::getFlaRemapTbl(arch),
                                             FMDeviceProperty::getFlaRemapIndexFromTargetId(arch, gpuTargetId) + i);
        }

        //
        // Do not ilwalidate SPA entry
        // because SPA is coming from the firmware it is not directly related to targetId
        //

        // Ilwalidate RID and RLAN entries
        mIlwalidIngressRidEntries.insert( gpuTargetId );
        mIlwalidIngressRlanEntries.insert( gpuTargetId );
    }
}

void
FMFabricParser::disableGpus ( std::set<GpuKeyType> &gpus, lwSwitchArchType arch )
{
    std::set<GpuKeyType>::iterator it;
    for ( it = gpus.begin(); it != gpus.end(); it++ )
    {
        GpuKeyType key = *it;
        disableGpu( key, arch );
    }
}

bool trunkPortSort( const trunkPort &a, const trunkPort &b )
{
    return  (a.has_localportnum() && b.has_localportnum() &&
             (a.localportnum() < b.localportnum()));
}

/*
 * prune a trunk port from the topology
 * used when a switch connected to a trunk port is pruned
 */
void
FMFabricParser::pruneOneTrunkPort ( uint32_t nodeId, uint32_t physicalId,
                                      lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        FM_LOG_DEBUG("pruneOneTrunkPort - Invalid lwswitch.");
        return;
    }

    FM_LOG_DEBUG("removing LWSwitch trunk port %d/%d/%d from routing configuration",
                 nodeId, physicalId, localportNum);

    for ( i = 0; i < pLwswitch->trunk_size(); i++ )
    {
        const trunkPort &trunk = pLwswitch->trunk(i);

        if ( trunk.has_localportnum() && trunk.localportnum() == localportNum )
            break;
    }

    if ( i >= pLwswitch->trunk_size() )
    {
        // trunk port with the specified localportNum is not found
        FM_LOG_WARNING("unable to find specified trunk port in topology for " NODE_ID_LOG_STR " %d LWSwitch physical id %d trunk port %d",
                       nodeId, physicalId, localportNum);
        return;
    }

    google::protobuf::RepeatedPtrField<trunkPort> *ports = pLwswitch->mutable_trunk();

    if ( i < pLwswitch->trunk_size() - 1 )
    {
        ports->SwapElements(i, pLwswitch->trunk_size() - 1);
    }
    ports->RemoveLast();

    // Reorder the list by local port number
    std::sort(ports->begin(), ports->end(), trunkPortSort);

    // remove the port from the config
    PortKeyType key;
    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex = localportNum;

    std::map <PortKeyType, lwswitch::switchPortInfo *>::iterator it;
    it = portInfo.find( key );
    if ( it != portInfo.end() )
    {
        FM_LOG_DEBUG("removed LWSwitch trunk port %d/%d/%d from routing configuration",
                    key.nodeId, key.physicalId, key.portIndex);
        portInfo.erase( it );
    }
}

/*
 * prune all trunk ports that are directly connected to a pruned Switch
 */
void
FMFabricParser::pruneTrunkPorts( void )
{
    int j, n, w;
    uint32_t i, physicalId, nodeId;
    trunkPort *port = NULL;

    if ( mDisabledSwitches.size() == 0 )
        return;

    for ( n = 0; n < mpFabric->fabricnode_size(); n++ )
    {
       const node &fnode = mpFabric->fabricnode(n);
       nodeId = fnode.has_nodeid() ? fnode.nodeid() : n;

       for ( w = 0; w < fnode.lwswitch_size(); w++ )
       {
           lwSwitch *pLwswitch = (lwSwitch *)&fnode.lwswitch(w);

           physicalId = w;
           if ( pLwswitch->has_physicalid() )
           {
               physicalId = pLwswitch->physicalid();
           }

           for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
           {
               for ( j = 0; j < pLwswitch->trunk_size(); j++ )
               {
                   port = (trunkPort *) &pLwswitch->trunk(j);
                   if ( port->has_localportnum() && ( port->localportnum() == i ) )
                   {
                       break;
                   }
                   else
                   {
                       port = NULL;
                   }
               }

               if ( !port )
               {
                   continue;
               }

               // find all trunk ports that are connected to a disabled switch
               // that is the trunk port farswitchid is the physicalId of a disabled switch
               if ( port->has_farswitchid() && port->has_farnodeid() )
               {
                   SwitchKeyType switchKey;
                   switchKey.nodeId = port->farnodeid();
                   switchKey.physicalId = port->farswitchid();

                   if ( mDisabledSwitches.find(switchKey) != mDisabledSwitches.end() )
                   {
                       pruneOneTrunkPort( nodeId, physicalId, pLwswitch, port->localportnum() );
                   }
               }
           }
       }
    }
}

bool switchSort( const lwSwitch& a, const lwSwitch& b )
{
    return ( ( a.has_physicalid() && b.has_physicalid() ) &&
             ( a.physicalid() < b.physicalid() ) );
}

void
FMFabricParser::pruneOneSwitch ( SwitchKeyType &key )
{
    int i;

    if ( key.nodeId > (uint32)mpFabric->fabricnode_size() )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " while trying to remove an LWSwitch from routing configration");
        return;
    }

    FM_LOG_INFO("removing LWSwitch " NODE_ID_LOG_STR " %d with physical id %d from routing configuration", key.nodeId, key.physicalId);

    node *pNode = (node *)&mpFabric->fabricnode(key.nodeId);
    for ( i = 0; i < pNode->lwswitch_size(); i++ )
    {
        const lwSwitch &lwswitch = pNode->lwswitch(i);

        if ( lwswitch.has_physicalid() &&
             ( lwswitch.physicalid() == key.physicalId ) )
        {
            break;
        }
    }

    if ( i >= pNode->lwswitch_size() )
    {
        // Switch with the specified key is not found
        return;
    }

    google::protobuf::RepeatedPtrField<lwSwitch> *lwswitchs = pNode->mutable_lwswitch();

    if ( i < pNode->lwswitch_size() - 1 )
    {
        lwswitchs->SwapElements(i, pNode->lwswitch_size() - 1);
    }
    lwswitchs->RemoveLast();

    // Reorder the list by physicalId
    std::sort(lwswitchs->begin(), lwswitchs->end(), switchSort);

    // remove the Switch from the config
    std::map <SwitchKeyType, lwswitch::switchInfo * >::iterator it;
    it = lwswitchCfg.find( key );
    if ( it != lwswitchCfg.end() )
    {
        FM_LOG_INFO("removed LWSwitch " NODE_ID_LOG_STR " %d physical id %d from routing configuration", key.nodeId, key.physicalId);
        lwswitchCfg.erase( it );
    }
}

void
FMFabricParser::pruneSwitches( void )
{
    std::set<SwitchKeyType>::iterator it;

    for ( it = mDisabledSwitches.begin(); it != mDisabledSwitches.end(); it++ )
    {
        SwitchKeyType key = *it;
        pruneOneSwitch( key );
    }
}

/*
 * add a Switch key to mDisabledSwitches
 */
void
FMFabricParser::disableSwitch ( SwitchKeyType &key )
{
    mDisabledSwitches.insert(key);
}

void
FMFabricParser::disableSwitches ( std::set<SwitchKeyType> &switches )
{
    std::set<SwitchKeyType>::iterator it;
    for (it = switches.begin(); it != switches.end(); it++ )
    {
        SwitchKeyType key = *it;
        mDisabledSwitches.insert(key);
    }
}

void
FMFabricParser::modifyFabric( PruneFabricType pruneType, lwSwitchArchType arch )
{
    switch ( pruneType )
    {
        case PRUNE_SWITCH:
        {
            pruneSwitches();
            pruneTrunkPorts();
            break;
        }
        case PRUNE_GPU:
        {
            pruneGpus();
            pruneAccessPorts();
            modifyRoutingTable(arch);
            break;
        }
        case PRUNE_PARTITION:
        {
            pruneSharedLWSwitchPartitions();
            break;
        }

        default:
        {
            FM_LOG_ERROR("unknown object type: %d to remove from fabric topology file", pruneType);
            return;
        }
    }

// this is to verify the running topology after all the prune operation.
// not required for production.
#ifdef DEBUG
    ofstream  outFile;
    outFile.open( DEFAULT_RUNNING_TOPOLOGY_FILE, ios::binary );

    if ( outFile.is_open() == false )
    {
        FM_LOG_ERROR("Failed to open output file %s, error is %s.",
                     DEFAULT_RUNNING_TOPOLOGY_FILE, strerror(errno));
        return;
    }

    // write the binary topology file
    int   fileLength = mpFabric->ByteSize();
    int   bytesWritten;
    char *bufToWrite = new char[fileLength];

    if ( bufToWrite == NULL )
    {
        outFile.close();
        return;
    }

    mpFabric->SerializeToArray( bufToWrite, fileLength );
    outFile.write( bufToWrite, fileLength );

    delete[] bufToWrite;
    outFile.close();
#endif

}

int
FMFabricParser::getNumDisabledGpus( void )
{
    return mDisabledGpuEndpointIds.size();
}

int
FMFabricParser::getNumDisabledSwitches( void )
{
    return mDisabledSwitches.size();
}

int
FMFabricParser::getNumDisabledPartitions( void )
{
    return mDisabledPartitions.size();
}

void
FMFabricParser::fabricParserCleanup( void )
{
    // TODO
}

bool
FMFabricParser::isSwtichGpioPresent( void )
{
    if ( mpFabric->fabricnode_size() > 0 )
    {
        const node &node = mpFabric->fabricnode(0);
        if (node.lwswitch_size() > 0)
        {
            const lwSwitch &lwswitch = node.lwswitch(0);
            if ( lwswitch.has_physicalid() )
            {
                return true;
            }
        }
    }

    return false;
}



// given nodeId, return nodeIndex
uint32_t
FMFabricParser::getNodeIndex(uint32_t nodeId)
{
    uint32_t nodeIndex = nodeId;

    for (uint32_t n = 0; n < (uint32_t)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        {
            return n;
        }
    }

    return nodeIndex;
}

#if LWCFG(GLOBAL_FEATURE_RID72837_KT_MULTINODE)
uint32_t
FMFabricParser::getMaxNumNodes()
{
    return mpFabric->fabricnode_size();
}
#endif

lwSwitchArchType
FMFabricParser::getFabricArch()
{
    if( mpFabric->has_arch() )
    {
        FM_LOG_DEBUG( "arch = %d", mpFabric->arch() );
        return mpFabric->arch();
    }
    else
    {
        return LWSWITCH_ARCH_TYPE_ILWALID;
    }
}

accessPort *
FMFabricParser::getAccessPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    accessPort *port = NULL;

    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        //TODO: Needs to be uncommented in future as this check is needed
        //if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        //{
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.access_size(); p++ )
                    {
                        const accessPort &access = lwswitch.access(p);
                        if ( access.has_localportnum() &&
                             ( access.localportnum() == portNum ) )
                        {
                            port = (accessPort *)&access;
                            return port;
                        }
                    }
                }
            }
        //}
    }

    return port;
}

accessPort *
FMFabricParser::getAccessPortInfoFromCopy( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    accessPort *port = NULL;
    for (int n = 0; n < (int)mpFabricCopy->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabricCopy->fabricnode(n);
        //TODO: Needs to be uncommented in future as this check is needed
        //if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        //{
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.access_size(); p++ )
                    {
                        const accessPort &access = lwswitch.access(p);
                        if ( access.has_localportnum() &&
                             ( access.localportnum() == portNum ) )
                        {
                            port = (accessPort *)&access;
                            return port;
                        }
                    }
                }
            }
        //}
    }

    return port;
}

trunkPort *
FMFabricParser::getTrunkPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    trunkPort *port = NULL;
    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        //TODO: Needs to be uncommented in future as this check is needed for multi-node
        //if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        //{
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.trunk_size(); p++ )
                    {
                        const trunkPort &trunk = lwswitch.trunk(p);
                        if ( trunk.has_localportnum() &&
                             ( trunk.localportnum() == portNum ) )
                        {
                            port = (trunkPort *)&trunk;
                            return port;
                        }
                    }
                }
            }
        //}
    }

    return port;
}

trunkPort *
FMFabricParser::getTrunkPortInfoFromCopy( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    trunkPort *port = NULL;
    for (int n = 0; n < (int)mpFabricCopy->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabricCopy->fabricnode(n);
        //TODO: Needs to be uncommented in future as this check is needed for multi-node
        //if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        //{
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.trunk_size(); p++ )
                    {
                        const trunkPort &trunk = lwswitch.trunk(p);
                        if ( trunk.has_localportnum() &&
                             ( trunk.localportnum() == portNum ) )
                        {
                            port = (trunkPort *)&trunk;
                            return port;
                        }
                    }
                }
            }
        //}
    }

    return port;
}

bool
FMFabricParser::getSwitchPortConfig( uint32_t nodeId, uint32_t physicalId,
                                     uint32_t portNum, switchPortConfig &portCfg )
{
    PortKeyType key;
    key.nodeId = nodeId;
    key.physicalId = physicalId;
    key.portIndex = portNum;

    std::map <PortKeyType, lwswitch::switchPortInfo *>::iterator it = portInfo.find(key);
    if ( it == portInfo.end() )
    {
        return false;
    }

    lwswitch::switchPortInfo *port = it->second;
    if ( port->has_config() )
    {
        portCfg = port->config();
        return true;
    }

    return false;
}

bool
FMFabricParser::isAccessPort( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    switchPortConfig portCfg;
    if ( getSwitchPortConfig( nodeId, physicalId, portNum, portCfg ) == false )
    {
        return false;
    }

    if ( portCfg.has_type() && (portCfg.type() != TRUNK_PORT_SWITCH) )
    {
        return true;
    }

    return false;
}

bool
FMFabricParser::isTrunkPort( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    switchPortConfig portCfg;
    if ( getSwitchPortConfig( nodeId, physicalId, portNum, portCfg ) == false )
    {
        return false;
    }

    if ( portCfg.has_type() && (portCfg.type() == TRUNK_PORT_SWITCH) )
    {
        return true;
    }

    return false;
}

const char *
FMFabricParser::getPlatformId()
{
    if ( mpFabric->has_name() )
    {
        return mpFabric->name().c_str();
    }
    else
    {
        return NULL;
    }
}

// get all switches that are connected to the specified switch
void
FMFabricParser::getConnectedSwitches( SwitchKeyType &key, std::map <SwitchKeyType, uint64_t> &connectedSwitches )
{
    connectedSwitches.clear();

    for (int n = 0; n < (int)mpFabricCopy->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabricCopy->fabricnode(n);

        for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
        {
            const lwSwitch &lwswitch = fnode.lwswitch(w);

            for (int p = 0; p < (int)lwswitch.trunk_size(); p++ )
            {
                const trunkPort &trunk = lwswitch.trunk(p);
                if ( !trunk.has_farnodeid() || trunk.farnodeid() != key.nodeId )
                    continue;

                if ( !trunk.has_farswitchid() || trunk.farswitchid() != key.physicalId )
                    continue;

                // found a trunk port that is connected to the specified switch
                SwitchKeyType switchKey;
                switchKey.nodeId = fnode.nodeid();
                switchKey.physicalId = lwswitch.has_physicalid() ? lwswitch.physicalid() : w;

                std::map <SwitchKeyType, uint64_t>::iterator it;
                it = connectedSwitches.find(switchKey);
                uint64_t portMask = 0;

                if ( it == connectedSwitches.end() )
                {
                    // the switch is not in the map, insert the connected switch
                    connectedSwitches.insert(make_pair(switchKey, portMask));
                }
                else
                {
                    // the switch is already in the map, get the portMask
                    portMask = it->second;
                }

                // update portMask by adding this trunk port
                uint32_t portNum = trunk.has_localportnum() ? trunk.localportnum() : p;
                portMask |= 1 << portNum;
                connectedSwitches[switchKey] = portMask;
            }
        }
    }
}

// get all GPUs and their port mask that are connected to the specified switch
void
FMFabricParser::getConnectedGpus( SwitchKeyType &key, std::map <GpuKeyType, uint64_t> &connectedGpus )
{
    connectedGpus.clear();
    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);

        for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
        {
            const lwSwitch &lwswitch = fnode.lwswitch(w);
            for (int p = 0; p < (int)lwswitch.access_size(); p++ )
            {
                const accessPort &access = lwswitch.access(p);
                if ( !access.has_farpeerid() || !access.has_farportnum() )
                    continue;

                GpuKeyType gpuKey;
                gpuKey.nodeId = fnode.nodeid();
                gpuKey.physicalId = access.farpeerid();

                std::map <GpuKeyType, uint64_t>::iterator it;
                it = connectedGpus.find(gpuKey);
                uint64_t portMask = 0;

                if ( it == connectedGpus.end() )
                {
                    // the gpu is not in the map, insert the connected switch
                    connectedGpus.insert(make_pair(gpuKey, portMask));
                }
                else
                {
                    // the GPU is already in the map, get the portMask
                    portMask = it->second;
                }

                // update portMask by adding this gpu port
                uint32_t portNum = access.farportnum();
                portMask |= 1 << portNum;
                connectedGpus[gpuKey] = portMask;
            }
        }
    }
}

// get all switches and their port mask that are connected to the specified GPU
void
FMFabricParser::getSwitchPortMaskToGpu( GpuKeyType &gpuKey, std::map <SwitchKeyType, uint64_t> &connectedSwitches )
{
    connectedSwitches.clear();

    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);

        for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
        {
            const lwSwitch &lwswitch = fnode.lwswitch(w);

            SwitchKeyType switchKey;
            switchKey.nodeId = fnode.nodeid();
            switchKey.physicalId = lwswitch.has_physicalid() ? lwswitch.physicalid() : w;

            for (int p = 0; p < (int)lwswitch.access_size(); p++ )
            {
                const accessPort &access = lwswitch.access(p);
                if ( !access.has_farpeerid() || !access.has_localportnum() ||
                     ( access.farpeerid() != gpuKey.physicalId ) )
                {
                    // switch port is not connected to the specified GPU
                    continue;
                }

                std::map <SwitchKeyType, uint64_t>::iterator it;
                it = connectedSwitches.find(switchKey);
                uint64_t portMask = 0;

                if ( it == connectedSwitches.end() )
                {
                    // the switch is not in the map, insert the connected switch
                    connectedSwitches.insert(make_pair(switchKey, portMask));
                }
                else
                {
                    // the switch is already in the map, get the portMask
                    portMask = it->second;
                }

                // update portMask by adding this access port
                uint32_t portNum = access.localportnum();
                portMask |= 1LL << portNum;
                connectedSwitches[switchKey] = portMask;
            }
        }
    }
}

// get all shared LWSwitch partitions that are using the specified switch
void
FMFabricParser::getSharedLWSwitchPartitionsWithSwitch( SwitchKeyType &key, std::set<PartitionKeyType> &partitions)
{
    partitions.clear();
    PartitionKeyType partition;
    partition.nodeId = key.nodeId;

    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);

        //TODO: Needs to be uncommented in future as this check is needed in future
        // if ( !fnode.has_nodeid() || ( fnode.nodeid() != key.nodeId ) ||
        //      !fnode.has_partitioninfo() )
        if (!fnode.has_partitioninfo()) {
            continue;
        }

        for (int p = 0; p < fnode.partitioninfo().sharedlwswitchinfo_size(); p++ )
        {
            sharedLWSwitchPartitionInfo partInfo = fnode.partitioninfo().sharedlwswitchinfo(p);

            for (int s = 0; s < partInfo.switchinfo_size(); s++ )
            {
                sharedLWSwitchPartitionSwitchInfo switchInfo = partInfo.switchinfo(s);
                if ( switchInfo.physicalid() != key.physicalId )
                    continue;

                // found a partition that is using the specified switch
                // add it to the set.
                partition.partitionId = partInfo.partitionid();
                partitions.insert(partition);
            }
        }
    }
}

// get all shared LWSwitch partitions that are using the specified GPU
void
FMFabricParser::getSharedLWSwitchPartitionsWithGpu( GpuKeyType &key, std::set<PartitionKeyType> &partitions)
{
    partitions.clear();

    PartitionKeyType partition;
    partition.nodeId = key.nodeId;

    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);

        //TODO: Needs to be uncommented in future as this check is needed
        // if ( !fnode.has_nodeid() || ( fnode.nodeid() != key.nodeId ) ||
        //      !fnode.has_partitioninfo() )
        if (!fnode.has_partitioninfo()) {
            continue;
        }

        for (int p = 0; p < fnode.partitioninfo().sharedlwswitchinfo_size(); p++ )
        {
            sharedLWSwitchPartitionInfo partInfo = fnode.partitioninfo().sharedlwswitchinfo(p);

            for (int s = 0; s < partInfo.gpuinfo_size(); s++ )
            {
                sharedLWSwitchPartitionGpuInfo gpuInfo = partInfo.gpuinfo(s);
                if ( gpuInfo.physicalid() != key.physicalId )
                    continue;

                // found a partition that is using the specified GPU
                // add it to the set.
                partition.partitionId = partInfo.partitionid();
                partitions.insert(partition);
            }
        }
    }
}

// get all shared LWSwitch partitions that are using trunk links
void
FMFabricParser::getSharedLWSwitchPartitionsWithTrunkLinks( uint32_t nodeId, std::set<PartitionKeyType> &partitions )
{
    partitions.clear();
    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++) {
        node pNode = mpFabric->fabricnode(n);

        //TODO: Needs to be uncommented in future as this check is needed
        // if ( !pNode->has_nodeid() || ( pNode->nodeid() != nodeId ) ||
        //      !pNode->has_partitioninfo() )
        if (!pNode.has_partitioninfo()) {
            continue;
        }

        for (int p = 0; p < pNode.partitioninfo().sharedlwswitchinfo_size(); p++ )
        {
            sharedLWSwitchPartitionInfo partInfo = pNode.partitioninfo().sharedlwswitchinfo(p);

            partitionMetaDataInfo pMetadata = partInfo.metadata();
            if (pMetadata.lwlinkintratrunkconncount() + pMetadata.lwlinkintertrunkconncount() > 0) {
                PartitionKeyType pKey;
                pKey.nodeId = n;
                pKey.partitionId = partInfo.partitionid();
                partitions.insert(pKey);
            }
        }
    }
}

bool sharedLWSwitchPartitionSort( const sharedLWSwitchPartitionInfo& a, const sharedLWSwitchPartitionInfo& b )
{
    return ( ( a.has_partitionid() && b.has_partitionid() ) &&
             ( a.partitionid() < b.partitionid() ) );
}

void
FMFabricParser::pruneOneSharedLWSwitchPartition( uint32_t nodeId, uint32_t partitionId )
{
    int i;
    NodeKeyType nodeKey;
    nodeKey.nodeId = nodeId;
    std::map <NodeKeyType, NodeConfig *>::iterator sit = NodeCfg.find(nodeKey);
    NodeConfig *cfg = (sit->second);

    nodeSystemPartitionInfo *partitionInfo = (nodeSystemPartitionInfo *)&(cfg->partitionInfo);
    for (i = 0; i < partitionInfo->sharedlwswitchinfo_size(); i++ )
    {
        const sharedLWSwitchPartitionInfo &partInfo = partitionInfo->sharedlwswitchinfo(i);
        if ( partInfo.has_partitionid() && ( partInfo.partitionid() == partitionId ) )
        {
            break;
        }
    }

    if ( i >= partitionInfo->sharedlwswitchinfo_size() )
    {
        // partition with specific partitionId is not found
        return;
    }

    // the disabled partition and reason is already logged by degraded manager.

    google::protobuf::RepeatedPtrField<sharedLWSwitchPartitionInfo> *partitions = partitionInfo->mutable_sharedlwswitchinfo();

    if ( i < partitionInfo->sharedlwswitchinfo_size() - 1 )
    {
        partitions->SwapElements(i, partitionInfo->sharedlwswitchinfo_size() - 1);
    }
    partitions->RemoveLast();

    // Reorder the list by partitionIdd
    std::sort(partitions->begin(), partitions->end(), sharedLWSwitchPartitionSort);
}

void
FMFabricParser::pruneSharedLWSwitchPartitions( void )
{
    std::set<PartitionKeyType>::iterator it;

    for ( it = mDisabledPartitions.begin(); it != mDisabledPartitions.end(); it++ )
    {
        PartitionKeyType key = *it;
        pruneOneSharedLWSwitchPartition( key.nodeId, key.partitionId );
    }
}

/*
 * add a Partition key to mDisabledPartitions
 */
void
FMFabricParser::disablePartition( PartitionKeyType &key )
{
    mDisabledPartitions.insert(key);
}

void
FMFabricParser::disablePartitions ( std::set<PartitionKeyType> &partitions )
{
    std::set<PartitionKeyType>::iterator it;
    for ( it = partitions.begin(); it != partitions.end(); it++ )
    {
        PartitionKeyType key = *it;
        disablePartition( key );
    }
    pruneSharedLWSwitchPartitions();
}

uint32_t
FMFabricParser::getNumGpusInPartition(PartitionKeyType key)
{
    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        //TODO: Needs to be uncommented in future as this check is needed
        // if ( !fnode.has_nodeid() || ( fnode.nodeid() != key.nodeId ) ||
        //      !fnode.has_partitioninfo() )
        if (!fnode.has_partitioninfo())
        {
            continue;
        }

        for (int p = 0; p < fnode.partitioninfo().sharedlwswitchinfo_size(); p++ )
        {
            sharedLWSwitchPartitionInfo partInfo = fnode.partitioninfo().sharedlwswitchinfo(p);

            if ( !partInfo.has_partitionid() ||
                 ( partInfo.partitionid() != key.partitionId ) )
            {
                continue;
            }

            return partInfo.gpuinfo_size();
        }
    }

    return 0;
}

bool switchInfoSort( const sharedLWSwitchPartitionSwitchInfo& a, const sharedLWSwitchPartitionSwitchInfo& b )
{
    return ( ( a.has_physicalid() && b.has_physicalid() ) &&
             ( a.physicalid() < b.physicalid() ) );
}

void
FMFabricParser::removeOneSwitchFromSharedPartition( SwitchKeyType &key,  sharedLWSwitchPartitionInfo *partInfo )
{
    int i;

    if ( key.nodeId > (uint32)mpFabric->fabricnode_size() )
    {
        FM_LOG_ERROR("invalid " NODE_ID_LOG_STR " while trying to remove an LWSwitch from routing configration");
        return;
    }

    FM_LOG_INFO("removing LWSwitch " NODE_ID_LOG_STR " %d physical id %d from shared LWSwitch partition id %d",
                key.nodeId, key.physicalId, partInfo->has_partitionid() ? partInfo->partitionid() : 0);

    for ( i = 0; i < partInfo->switchinfo_size(); i++ )
    {
        const sharedLWSwitchPartitionSwitchInfo &switchInfo = partInfo->switchinfo(i);

        if ( switchInfo.has_physicalid() &&
             ( switchInfo.physicalid() == key.physicalId ) )
        {
            break;
        }
    }

    if ( i >= partInfo->switchinfo_size() )
    {
        // Switch with the specified key is not found
        return;
    }

    google::protobuf::RepeatedPtrField<sharedLWSwitchPartitionSwitchInfo> *switchInfoes = partInfo->mutable_switchinfo();

    if ( i < partInfo->switchinfo_size() - 1 )
    {
        switchInfoes->SwapElements(i, partInfo->switchinfo_size() - 1);
    }
    switchInfoes->RemoveLast();

    // Reorder the list by physicalId
    std::sort(switchInfoes->begin(), switchInfoes->end(), switchInfoSort);
}

// update all shared LWSwitch Partition
// - remove the switchInfo
// - update GPU numEnabledLinks and numEnabledMask if it is connected to the removed switch
void
FMFabricParser::removeSwitchFromSharedLWSwitchPartitions( SwitchKeyType &key, std::map<uint32_t, int> &modifiedPartitions )
{
    // lwrrently look for only node 0.
    // TODO: re-visit during multi-node support
    // TODO: also add checks for whether the system has partitionInfo
    uint32_t nodeId = 0;
    std::map <NodeKeyType, NodeConfig *>::iterator sit = NodeCfg.begin();
    NodeConfig *cfg = (sit->second);
    nodeSystemPartitionInfo *partitionInfo = &(cfg->partitionInfo);
    for (int p = 0; p < partitionInfo->sharedlwswitchinfo_size(); p++ )
    {
        sharedLWSwitchPartitionInfo *partInfo = (sharedLWSwitchPartitionInfo *)&(partitionInfo->sharedlwswitchinfo(p));
        partitionMetaDataInfo *partMetaData = (partitionMetaDataInfo*)&(partInfo->metadata());

        for ( int s = 0; s < partInfo->switchinfo_size(); s++ )
        {
            sharedLWSwitchPartitionSwitchInfo switchInfo = partInfo->switchinfo(s);
            if ( !switchInfo.has_physicalid() || ( switchInfo.physicalid() != key.physicalId ) )
            {
                continue;
            }

            if ( partMetaData->lwlinkintratrunkconncount())
            {
                uint64 trunkLinkMask = 0;
                uint32_t trunkLinkCount = 0;
                //TODO: change for multinode, lwrrently works for 1 node
                getSwitchTrunkLinkMask(nodeId, key.physicalId, trunkLinkMask);
                while (trunkLinkMask) {
                    trunkLinkCount += trunkLinkMask & 1;
                    trunkLinkMask >>= 1;
                }
                // this function is called twice per switch pair and we would subtract the trunklinks twice for each pair
                // hence we divide it by 2, so that overall, we just subtract trunklink once for each pair
                uint32_t lwLinkIntraTrunkConnCount = partMetaData->lwlinkintratrunkconncount() - (trunkLinkCount/2);
                partMetaData->set_lwlinkintratrunkconncount(lwLinkIntraTrunkConnCount);
            }

            uint64_t switchEnabledLinkMask = switchInfo.enabledlinkmask();
            // the switch is used in this partition, remove it
            removeOneSwitchFromSharedPartition( key,  partInfo );

            if (modifiedPartitions.find(p) == modifiedPartitions.end()) {
                modifiedPartitions[p] = 0;
            }

            // the switch is removed from the partition
            // update numEnabledLinks and numEnabledMask on all GPUs in the partition
            for (uint32_t portNum = 0; switchEnabledLinkMask != 0; portNum++, switchEnabledLinkMask = switchEnabledLinkMask >> 1)
            {
                accessPort *accessport = getAccessPortInfo( key.nodeId, key.physicalId, portNum );
                if ( accessport == NULL )
                {
                    accessport = getAccessPortInfoFromCopy( key.nodeId, key.physicalId, portNum);
                    if (accessport == NULL) {
                        continue;
                    }
                }

                // update GPU numEnabledLinks and numEnabledMask
                for ( int g = 0; g < partInfo->gpuinfo_size(); g++ )
                {
                    sharedLWSwitchPartitionGpuInfo *gpuInfo = (sharedLWSwitchPartitionGpuInfo *)&partInfo->gpuinfo(g);

                    // the access port is connected to a GPU in the partition
                    if ( accessport->has_farpeerid() &&
                         ( accessport->farpeerid() == gpuInfo->physicalid() ) )
                    {
                        uint32_t gpuPortNum = accessport->farportnum();
                        uint32_t gpuNumEnabledLinks = gpuInfo->numenabledlinks() - 1;
                        uint64_t gpuEnabledLinkMask = gpuInfo->enabledlinkmask() & (~(1LL << gpuPortNum));

                        gpuInfo->set_numenabledlinks(gpuNumEnabledLinks);
                        gpuInfo->set_enabledlinkmask(gpuEnabledLinkMask);
                        modifiedPartitions[p] = 1;
                    }
                }
            }
        }
    }
}

uint32_t
FMFabricParser::getNumConfiguredSharedLwswitchPartition( uint32_t nodeId )
{
    uint32_t count = 0;
    std::map <PartitionKeyType, sharedLWSwitchPartitionInfo *>::iterator it;

    for ( it = sharedLwswitchPartitionCfg.begin(); it != sharedLwswitchPartitionCfg.end(); it++ )
    {
        PartitionKeyType key = it->first;
        if ( key.nodeId == nodeId )
        {
            count++;
        }
    }
    return count;
}

sharedLWSwitchPartitionInfo *
FMFabricParser::getSharedLwswitchPartitionCfg( uint32_t nodeId, uint32_t partitionId )
{
    sharedLWSwitchPartitionInfo *partInfo = NULL;

    PartitionKeyType key;
    key.nodeId = nodeId;
    key.partitionId = partitionId;

    std::map <PartitionKeyType, sharedLWSwitchPartitionInfo *>::iterator it;
    it = sharedLwswitchPartitionCfg.find(key);

    if ( it != sharedLwswitchPartitionCfg.end() )
    {
        partInfo = it->second;
    }

    return partInfo;
}

// return true if a trunk port is connected to the specified switch,
//             and the portNum on the switch
//
// return false if a trunk port is not connect to the specified switch
bool
FMFabricParser::isTrunkConnectedToSwitch( PortKeyType &trunkPortKey,
                                          SwitchKeyType &switchKey,
                                          uint32_t &portNum)
{
    trunkPort *trunkPortInfo = getTrunkPortInfo(trunkPortKey.nodeId,
                                                trunkPortKey.physicalId,
                                                trunkPortKey.portIndex);

    // Test if the trunk port on the peer switch is connected to this switch
    // In multi node systems, not all trunk ports on one switch are connect to the
    // same peer switch
    if (!trunkPortInfo || !trunkPortInfo->has_farnodeid() ||
        (trunkPortInfo->farnodeid() != switchKey.nodeId) || !trunkPortInfo->has_farswitchid() ||
        (trunkPortInfo->farswitchid() != switchKey.physicalId) || !trunkPortInfo->has_farportnum()) {
        // The trunk port is not connected to this switch
        return false;
    }

    portNum = trunkPortInfo->farportnum();
    return true;
}

// return true if a GPU specified by the key is found, and its targetId is returned
//        false if a GPU specified by the key is not found
bool
FMFabricParser::getGpuTargetIdFromKey( GpuKeyType key, uint32_t &targetId )
{
    std::map <GpuKeyType, lwswitch::gpuInfo *>::iterator it = gpuCfg.find(key);
    if ( it == gpuCfg.end() ) {
        return false;
    }

    lwswitch::gpuInfo *info = it->second;
    if ( info && info->has_targetid() ) {
        targetId = info->targetid();
        return true;
    }

    return false;
}

// return true if a GPU specified by the targetId is found, and its key is returned
//        false if a GPU specified by the targetId  is not found
bool
FMFabricParser::getGpuKeyFromTargetId( uint32_t targetId, GpuKeyType &key )
{
    std::map <uint32_t, GpuKeyType>::iterator it = mGpuTargetIdTbl.find(targetId);
    if ( it == mGpuTargetIdTbl.end() ) {
        return false;
    }

    key = it->second;
    return true;
}

void
FMFabricParser::ridEntryToDstPortMask( ridRouteEntry *ridEntry, uint64_t &dstPortMask )
{
    dstPortMask = 0;

    if ( !ridEntry ) {
        return;
    }

    for (int i = 0; i < ridEntry->portlist_size(); i++ )
    {
        routePortList port = ridEntry->portlist(i);
        uint32_t portNum = port.portindex();
        dstPortMask |= (uint64_t)1 << portNum;
    }
}


