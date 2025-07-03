#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>

#include "logging.h"
#include "DcgmFabricParser.h"
#include <g_lwconfig.h>


using namespace std;

DcgmFabricParser::DcgmFabricParser()
{
    mpFabric = new( fabric );
    lwLinkConnMap.clear();
};

DcgmFabricParser::~DcgmFabricParser()
{
    lwLinkConnMap.clear();

    fabricParserCleanup();

    if ( mpFabric )
    {
        delete( mpFabric );
    }
    mpFabric = NULL;
};

FM_ERROR_CODE
DcgmFabricParser::parseOneGpu( const GPU &gpu, uint32_t nodeId, int gpuIndex )
{
    lwswitch::gpuInfo *info;
    peerIDPortMap *pPeerIdMap;
    uint64_t addrBase, addrRange;
    GpuKeyType gpuKey;
    int physicalId = 0, gpuEndPointId = 0;

    // find the gpu PhysicalId from fabric address. 
    // In topology file, the base address is assigned as gpuEndPointId << 36, so do a reverse callwlation
    if ( gpu.has_fabricaddrbase() )
    {
        addrBase = gpu.fabricaddrbase();

        if ( gpu.has_physicalid() )
        {
            physicalId = gpu.physicalid();
        }
        else
        {
            gpuEndPointId = GPU_ENDPOINT_ID_FROM_ADDR_BASE(addrBase);
            physicalId = GPU_PHYSICAL_ID(gpuEndPointId);
        }
    }
    else
    {
        PRINT_ERROR("%d %d", "Missing fabric address base for nodeId %d, gpuIndex %d ",
                     nodeId, gpuIndex);
        return FM_ILWALID_GPU_CFG;
    }

    PRINT_DEBUG("%d %d %d", "nodeId %d, gpuIndex %d physicalId %d ",
                nodeId, gpuIndex, physicalId);
    
    info = new lwswitch::gpuInfo();
    gpuKey.nodeId = nodeId;
    gpuKey.physicalId  = physicalId;
    gpuCfg.insert(make_pair(gpuKey, info));

    info->set_gpuphysicalid( physicalId );

    addrRange = GPU_FABRIC_DEFAULT_ADDR_RANGE;
    if ( gpu.has_fabricaddrrange() )
    {
        addrRange = gpu.fabricaddrrange();
    }

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
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
#endif

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

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
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

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum,
                                            const ingressResponseTable &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    ingressResponseTable *entry;
    RespTableKeyType key;

    if ( !IS_INGR_RESP_VALID(index) )
    {
        PRINT_ERROR("%d %d %d %d",
                    "Invalid ingress response entry: nodeId %d lwswitch physicalId %d localPortNum %d Index %d.",
                    nodeId, physicalId, localPortNum, index);
        return FM_ILWALID_TABLE_ENTRY;
    }

    entry = new ingressResponseTable;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    respEntry.insert(make_pair(key, entry));

    return FM_SUCCESS;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

FM_ERROR_CODE
DcgmFabricParser::parseOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId,
                                              uint32_t localPortNum,
                                              const rmapPolicyEntry &cfg )
{
    int index = cfg.has_index() ? cfg.index() : -1;
    rmapPolicyEntry *entry;
    RmapTableKeyType key;

    if ( !LR_IS_INGR_RMAP_VALID(index) )
    {
        PRINT_ERROR("%d %d %d %d",
                    "Invalid ingress RMAP entry: nodeId %d lwSwitch physicalId %d localPortNum %d Index %d.",
                    nodeId, physicalId, localPortNum, index);
        return FM_ILWALID_TABLE_ENTRY;
    }



    entry = new rmapPolicyEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    rmapEntry.insert(make_pair(key, entry));

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId,
                                             uint32_t localPortNum,
                                             const ridRouteEntry &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    ridRouteEntry *entry;
    RidTableKeyType key;

    if ( !LR_IS_INGR_RID_VALID(index) )
    {
        PRINT_ERROR("%d %d %d %d",
                    "Invalid ingress RID entry: nodeId %d lwSwitch physicalId %d localPortNum %d Index %d.",
                    nodeId, physicalId, localPortNum, index);
        return FM_ILWALID_TABLE_ENTRY;
    }

    entry = new ridRouteEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    ridEntry.insert(make_pair(key, entry));

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId,
                                             uint32_t localPortNum,
                                             const rlanRouteEntry &cfg)
{
    int index = cfg.has_index() ? cfg.index() : -1;
    rlanRouteEntry *entry;
    RlanTableKeyType key;

    if ( !LR_IS_INGR_RLAN_VALID(index) )
    {
        PRINT_ERROR("%d %d %d %d",
                    "Invalid ingress RLAN entry: nodeId %d lwSwitch physicalId %d localPortNum %d Index %d.",
                    nodeId, physicalId, localPortNum, index);
        return FM_ILWALID_TABLE_ENTRY;
    }

    entry = new rlanRouteEntry;
    entry->CopyFrom( cfg );

    key.nodeId  = nodeId;
    key.physicalId = physicalId;
    key.portIndex  = localPortNum;
    key.index      = index;
    rlanEntry.insert(make_pair(key, entry));

    return FM_SUCCESS;
}

#endif

FM_ERROR_CODE
DcgmFabricParser::parseOneGangedLinkEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const gangedLinkTable &cfg )
{
    int32_t index, *data;
    GangedLinkTableKeyType key;

    if ( !IS_PORT_VALID(localPortNum) )
    {
        PRINT_ERROR("%d %d %d", "Invalid local port number: nodeId %d lwswitch physicalId %d localPortNum %d.",
                    nodeId, physicalId, localPortNum);
        return FM_ILWALID_GANGED_LINK_ENTRY;
    }

    if ( cfg.data_size() != GANGED_LINK_TABLE_SIZE )
    {
        PRINT_ERROR("%d %d %d %d", "Not all entries are specified, nodeId %d lwswitch physicalId %d localPortNum %d, size %d.",
                    nodeId, physicalId, localPortNum, cfg.data_size() );
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

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseIngressReqTable( const lwSwitch &lwswitch,
                                        uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);

        for ( i = 0; i < access.reqrte_size(); i++)
        {
            ec = parseOneIngressReqEntry( nodeId, physicalId,
                                     access.has_localportnum() ? access.localportnum() : port,
                                     access.reqrte(i) );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d", "Failed to parse ingress req entry on nodeId %d, lwswitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_ILWALID_PORT;
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

            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d",
                            "Failed to parse ingress req entry on nodeId %d, lwswitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_ILWALID_PORT;
            }
        }
    }
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseIngressRespTable( const lwSwitch &lwswitch,
                                         uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);
        for ( i = 0; i < access.rsprte_size(); i++)
        {
            ec = parseOneIngressRespEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.rsprte(i) );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d",
                            "Failed to parse ingress resp entry on nodeId %d, lwswitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_ILWALID_PORT;
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
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d", "Failed to parse ingress resp entry on nodeId %d, lwswitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_ILWALID_PORT;
            }
        }
    }
    return FM_SUCCESS;
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

FM_ERROR_CODE
DcgmFabricParser::parseIngressRmapTable( const lwSwitch &lwSwitch,
                                        uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwSwitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);

        for ( i = 0; i < access.rmappolicytable_size(); i++)
        {
            ec = parseOneIngressRmapEntry( nodeId, physicalId,
                                     access.has_localportnum() ? access.localportnum() : port,
                                     access.rmappolicytable(i) );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d", "Failed to parse ingress RMAP entry on nodeId %d, lwSwitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_ILWALID_PORT;
            }
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseIngressRidTable( const lwSwitch &lwSwitch,
                                         uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwSwitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);
        for ( i = 0; i < access.ridroutetable_size(); i++)
        {
            ec = parseOneIngressRidEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.ridroutetable(i) );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d",
                            "Failed to parse ingress RID entry on nodeId %d, lwSwitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_ILWALID_PORT;
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
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d", "Failed to parse ingress RID entry on nodeId %d, lwSwitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_ILWALID_PORT;
            }
        }
    }
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseIngressRlanTable( const lwSwitch &lwSwitch,
                                         uint32_t nodeId, uint32_t physicalId )
{
    int port, i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwSwitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwSwitch.access_size(); port++ )
    {
        const accessPort &access = lwSwitch.access(port);
        for ( i = 0; i < access.rlanroutetable_size(); i++)
        {
            ec = parseOneIngressRlanEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.rlanroutetable(i) );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d",
                            "Failed to parse ingress RLAN entry on nodeId %d, lwSwitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port, i);
                return FM_ILWALID_PORT;
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
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d %d", "Failed to parse ingress RLAN entry on nodeId %d, lwSwitch physicalId %d, localportnum %d, index %d.",
                            nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port, i);
                return FM_ILWALID_PORT;
            }
        }
    }
    return FM_SUCCESS;
}

#endif

FM_ERROR_CODE
DcgmFabricParser::parseGangedLinkTable( const lwSwitch &lwswitch,
                                        uint32_t nodeId, uint32_t physicalId )
{
    int port;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
                nodeId, physicalId);

    for ( port = 0; port < lwswitch.access_size(); port++ )
    {
        const accessPort &access = lwswitch.access(port);
        if ( access.has_gangedlinktbl() )
        {
            ec = parseOneGangedLinkEntry( nodeId, physicalId,
                                           access.has_localportnum() ? access.localportnum() : port,
                                           access.gangedlinktbl() );
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d",
                            "Failed to parse ganged link entry on nodeId %d, lwswitch physicalId %d, localportnum %d.",
                            nodeId, physicalId, access.has_localportnum() ? access.localportnum() : port);
                return FM_ILWALID_PORT;
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
            if ( ec != FM_SUCCESS )
            {
                PRINT_ERROR("%d %d %d", "Failed to parse ganged link entry on nodeId %d, lwswitch physicalId %d, localportnum %d.",
                            nodeId, physicalId, trunk.has_localportnum() ? trunk.localportnum() : port);
                return FM_ILWALID_PORT;
            }
        }
    }
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOnePort(const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId,
                               int portIndex, int isAccess)
{
    PortKeyType key;
    int localPortNum;
    lwswitch::switchPortInfo *info;
    switchPortConfig *cfg;
    PortType portType;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d %d %d", "nodeId %d lwswitch physicalId %d portIndex %d isAccess %d.",
                nodeId, physicalId, portIndex, isAccess);

    if ( !IS_PORT_VALID(portIndex) )
    {
        PRINT_ERROR("%d %d %d", "Invalid port index: nodeId %d, lwswitch physicalId %d, portIndex %d.",
                    nodeId, physicalId, portIndex);
        return FM_ILWALID_PORT;
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
        PRINT_ERROR("%d %d %d", "Invalid localPortNum nodeId %d, lwswitch physicalId %d localPortNum %d.",
                    nodeId, physicalId, localPortNum);
        return FM_ILWALID_PORT;
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
        PRINT_ERROR("%d %d %d", "Invalid port config nodeId %d, lwswitch physicalId %d localPortNum %d.",
                    nodeId, physicalId, localPortNum);
        return FM_ILWALID_PORT_CFG;
    }


    info->set_allocated_config( cfg );
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseAccessPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    int i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
                nodeId, physicalId);

    for ( i = 0; i < lwswitch.access_size(); i++ )
    {
        ec = parseOnePort(lwswitch, nodeId, physicalId, i, 1);
        if ( ec != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d", "Failed to parse port node %d lwswitch physicalId %d port %d with error %d.",
                        nodeId, physicalId, i, ec);
            return ec;
        }
    }
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseTrunkPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    int i;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
                nodeId, physicalId);

    for ( i = 0; i < lwswitch.trunk_size(); i++ )
    {
        ec = parseOnePort(lwswitch, nodeId, physicalId, i, 0);
        if ( ec != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d %d", "Failed to parse port node %d lwswitch physicalId %d port %d with error %d.",
                        nodeId, physicalId, i, ec);
            return ec;
        }
    }
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneLwswitch( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId )
{
    lwswitch::switchInfo *pInfo = new lwswitch::switchInfo;
    switchConfig *pConfig;
    int i;
    SwitchKeyType key;
    FM_ERROR_CODE ec;

    PRINT_DEBUG("%d %d", "nodeId %d, lwswitch physicalId %d ",
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
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse access ports on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;
    }

    // trunk ports
    ec = parseTrunkPorts( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse trunk ports on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;
    }

    // Ingress request table
    ec = parseIngressReqTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ingress request on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;
    }

    // Ingress response table
    ec = parseIngressRespTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ingress response on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;

    }
   
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)

    // Ingress remap/policy table
    ec = parseIngressRmapTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ingress remap on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;
    }

    // Ingress route ID table
    ec = parseIngressRidTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ingress route ID on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;

    }

    // Ingress route link aggregate number table
    ec = parseIngressRlanTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ingress route RLAN on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);
        return ec;

    }

    // Ganged link table
    ec = parseGangedLinkTable( lwswitch, nodeId, physicalId );
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d %d",
                    "Failed to parse ganged link table on node %d lwswitch physicalId %d with error %d.",
                    nodeId, physicalId, ec);

    }
#endif

    return ec;
}

int
DcgmFabricParser::checkLWLinkConnExists( TopologyLWLinkConnList &connList,
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
DcgmFabricParser::copyAccessPortConnInfo( accessPort accessPort, uint32_t nodeId,
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
DcgmFabricParser::copyTrunkPortConnInfo( trunkPort trunkPort, uint32_t nodeId,
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
}

FM_ERROR_CODE
DcgmFabricParser::parseLWLinkConnections( const node &node, uint32_t nodeId )
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
            copyAccessPortConnInfo(lwswitch.access(j), nodeId, physicalId, j, lwlinkConnList);
        }
        // parse trunk connections
        for ( int k = 0; k < lwswitch.trunk_size(); k++ )
        {
            copyTrunkPortConnInfo(lwswitch.trunk(k), nodeId, physicalId, k, lwlinkConnList);
        }
    }

    lwLinkConnMap.insert( std::make_pair(nodeId, lwlinkConnList) );
    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseOneNode( const node &node, int nodeIndex )
{
    int i;
    FM_ERROR_CODE ec;
    NodeKeyType key;
    NodeConfig *pNode;
    uint32_t nodeId = node.has_nodeid() ? node.nodeid() : nodeIndex;

    PRINT_DEBUG("%d %d", "fabricNode index %d, nodeId %d.", nodeIndex, nodeId);

    pNode = new NodeConfig;
    pNode->nodeId = nodeId;
    key.nodeId = nodeId;
    NodeCfg.insert(make_pair(key, pNode));

    if ( node.has_ipaddress() )
    {
        pNode->IPAddress = new std::string( node.ipaddress() );
    }
    else
    {
        pNode->IPAddress = new std::string( DCGM_HOME_IP );
    }

    pNode->partitionInfo.Clear();
    if ( node.has_partitioninfo() )
    {
        pNode->partitionInfo = node.partitioninfo();
    }

    //validate the total number of GPUs, and LWSwitches present in the topology file
    if ( node.gpu_size() > MAX_NUM_GPUS_PER_NODE )
    {
        PRINT_ERROR("", "Failed to parse topology: Number of GPUs to configure is more than allowed per node");
        return FM_ILWALID_GPU;
    }

    if ( node.lwswitch_size() > MAX_NUM_WILLOWS_PER_NODE )
    {
        PRINT_ERROR("", "Failed to parse topology: Number of LWSwitches to configure is more than allowed per node");
        return FM_ILWALID_WILLOW;
    }

    // configure all the GPUs
    for ( i = 0; i < node.gpu_size(); i++ )
    {
        ec = parseOneGpu( node.gpu(i), nodeId, i );
        if ( ec != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d", "Failed to parse node %d gpu %d with error %d.",
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
        }

        ec = parseOneLwswitch(lwswitch , nodeId, physicalId );
        if ( ec != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d %d", "Failed to parse node %d lwswitch %d with error %d.",
                        nodeId, i, ec);
            return ec;
        }
    }

    ec = parseLWLinkConnections(node, nodeId);
    if ( ec != FM_SUCCESS )
    {
        PRINT_ERROR("%d %d", "Failed to parse node %d lwlink connections with error %d.",
                    nodeId, ec);
        return ec;
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseFabricTopology( const char *topoFile )
{
    int i;
    FM_ERROR_CODE ec;

    if ( !topoFile )
    {
        PRINT_ERROR(" ", "Invalid topology file.");
        return FM_FILE_ILWALID;
    }

    // Read the protobuf binary file.
    fstream input(topoFile, ios::in | ios::binary);
    if ( !input )
    {
        PRINT_ERROR("%s", "Failed to open file %s.", topoFile);
        return FM_FILE_OPEN_ERR;
    }
    else if ( !mpFabric->ParseFromIstream(&input) )
    {
        PRINT_ERROR("%s", "Failed to parse file %s.", topoFile);
        input.close();
        return FM_FILE_PARSING_ERR;
    }

    input.close();
    PRINT_INFO("%s %s %s", "Parsed file %s successfully. topology name: %s, build time: %s.",
               topoFile,
               mpFabric->has_name() ? mpFabric->name().c_str() : "Not set",
               mpFabric->has_time() ? mpFabric->time().c_str() : "Not set");

    if ( mpFabric->fabricnode_size() > MAX_NUM_NODES )
    {
        PRINT_ERROR("", "Failed to parse topology: Number of Nodes to configure is more than allowed");
        return FM_ILWALID_NODE;
    }

    for (int i = 0; i < mpFabric->fabricnode_size(); i++)
    {
        ec = parseOneNode ( mpFabric->fabricnode(i), i );
        if ( ec != FM_SUCCESS )
        {
            PRINT_ERROR("%d %d", "Failed to parse node %d with error %d.",
                        i, ec);
            return ec;
        }
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseDisableGPUConf( std::vector<std::string> &gpus )
{
    int i, j, rc, nodeId, physicalId;
    GpuKeyType key;

    // i, starting from 1, as 0 is str --disable-gpu
    for ( i = 1; i < (int)gpus.size(); i++ )
    {
        rc = sscanf( gpus[i].c_str(), "%d/%d", &nodeId, &physicalId );
        if ( ( rc != 2 ) || !IS_NODE_VALID( nodeId ) || !IS_GPU_VALID( physicalId ) )
        {
            PRINT_ERROR("%d %d", "Invalid GPU %d/%x",
                        nodeId, physicalId);
            continue;
        }

        key.nodeId = nodeId;
        key.physicalId = physicalId;


        disableGpu( key );

        PRINT_INFO("%d %d", "GPU %d/%d is disabled.", nodeId, physicalId);
    }


    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseDisableSwitchConf( std::vector<std::string> &switches )
{
    int i, j, rc, nodeId, physicalId;
    SwitchKeyType key;

    // i, starting from 1, as 0 is str --disable-switch
    for ( i = 1; i < (int)switches.size(); i++ )
    {
        rc = sscanf( switches[i].c_str(), "%d/%x", &nodeId, &physicalId );
        if ( ( rc != 2 ) || !IS_NODE_VALID( nodeId ) || !IS_WILLOW_VALID( physicalId ) )
        {
            PRINT_ERROR("%d %d", "Invalid switch %d/%d",
                        nodeId, physicalId);
            continue;
        }

        key.nodeId = nodeId;
        key.physicalId = physicalId;
        disableSwitch( key );
        PRINT_INFO("%d %d", "Switch %d/%d is disabled.", nodeId, physicalId);
    }

    return FM_SUCCESS;
}

FM_ERROR_CODE
DcgmFabricParser::parseLoopbackPorts( std::vector<std::string> &ports )
{
    int i, rc, nodeId, physicalId, portIndex;
    PortKeyType key;

    // i, starting from 1, as 0 is str --loopback-port
    for ( i = 1; i < (int)ports.size(); i++ )
    {
        rc = sscanf( ports[i].c_str(), "%d/%d/%d", &nodeId, &physicalId, &portIndex);
        if ( ( rc != 3 ) || !IS_NODE_VALID( nodeId ) || !IS_PORT_VALID( portIndex ) )
        {
            PRINT_ERROR("%d %d %d", "Invalid port %d/%d/%d",
                        nodeId, physicalId, portIndex);
            continue;
        }

        key.nodeId  = nodeId;
        key.physicalId = physicalId;
        key.portIndex  = portIndex;

        mLoopbackPorts.insert(key);
        PRINT_INFO("%d %d %d", "Port %d/%d/%d is put to loopback.",
                   nodeId, physicalId, portIndex);
    }

    return FM_SUCCESS;
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
FM_ERROR_CODE
DcgmFabricParser::parseFabricTopologyConf( const char *topoConfFileName )
{
    uint i;
    FM_ERROR_CODE ec;

    if ( !topoConfFileName )
    {
        PRINT_ERROR(" ", "Invalid topology conf file.");
        return FM_FILE_ILWALID;
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
                parseDisableGPUConf(gpus);
            }
            else if ( line.find(OPT_DISABLE_SWITCH) == 0 )
            {
                std::istringstream str(line);
                std::vector<std::string> switches((std::istream_iterator<std::string>(str)),
                        std::istream_iterator<std::string>());
                parseDisableSwitchConf(switches);
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
                PRINT_ERROR("%s", "Unknown option %s.", line.c_str());
            }

        }
        inFile.close();
    }

    PRINT_DEBUG("%s", "Parsed file %s successfully.", topoConfFileName);
    return FM_SUCCESS;
}

/*
 * ilwalidate ingress request entry if the port is connected to a pruned GPU
 * make outgoing port to be the port itself, if the port is in loopback
 */
void
DcgmFabricParser::modifyOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId,
                                            uint32_t localPortNum, ingressRequestTable *entry )
{
    PortKeyType portKey;
    int index = entry->index();

    if ( mIlwalidIngressReqEntries.find(index) != mIlwalidIngressReqEntries.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
                    nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress request entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
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
DcgmFabricParser::modifyOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId,
                                             uint32_t localPortNum, ingressResponseTable *entry )
{
    PortKeyType portKey;
    int index = entry->index();

    if ( mIlwalidIngressRespEntries.find(index) != mIlwalidIngressRespEntries.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
                    nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress response entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;
    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "loopback nodeId %d, physicalId %d, localPortNum %d, index %d.",
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

void 
DcgmFabricParser::modifyOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                   rmapPolicyEntry *entry )
{
    PortKeyType portKey;
    int index = entry->index();
    int targetId; 
    if ( mIlwalidIngressRmapEntries.find(index) != mIlwalidIngressRmapEntries.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
                    nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RMAP entries will be set to invalid
        entry->set_entryvalid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
                    nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback, set the outgoing target ID to be its own GPU
        // Note that on LimeRock the target ID is effectively the GPA index until/unless
        // we get GPUs with > 64G ( NUM_INGR_RMAP_ENTRIES_PER_AMPERE = 1 )
        // for future-proofing we use target ID = index / NUM_INGR_RMAP_ENTRIES_PER_AMPERE 
        // The FLA target ID is  
        // ( index - LR_FIRST_FLA_RMAP_SLOT ) / NUM_INGR_RMAP_ENTRIES_PER_AMPERE

        targetId = index / NUM_INGR_RMAP_ENTRIES_PER_AMPERE;
        if ( targetId >= LR_FIRST_FLA_RMAP_SLOT ) 
        {
            targetId = ( index - LR_FIRST_FLA_RMAP_SLOT ) / NUM_INGR_RMAP_ENTRIES_PER_AMPERE;
        }

        entry->set_targetid(targetId);
      
    }

}

void 
DcgmFabricParser::modifyOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    ridRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID

    if ( mIlwalidIngressRidEntries.find(index) != mIlwalidIngressRidEntries.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
                    nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RID entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
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
DcgmFabricParser::modifyOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    rlanRouteEntry *entry )
{
    PortKeyType portKey;
    routePortList *list;
    int index = entry->index();
    // Note that on LimeRock the index is effectively the target ID 
    if ( mIlwalidIngressRlanEntries.find(index) != mIlwalidIngressRlanEntries.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "ilwalidate nodeId %d, physicalId %d, localPortNum %d, index %d.",
                    nodeId, physicalId, localPortNum, index);

        // If a GPU is pruned, its ingress RLAN entries will be set to invalid
        entry->set_valid(0);
    }

    portKey.nodeId  = nodeId;
    portKey.physicalId = physicalId;
    portKey.portIndex  = localPortNum;

    if ( mLoopbackPorts.find(portKey) !=  mLoopbackPorts.end() )
    {
        PRINT_DEBUG("%d %d %d %d",
                    "loopback nodeId %d, physicalId %d, localPortNum %d, index %d.\n",
                    nodeId, physicalId, localPortNum, index);

        // port is set to be in loopback. no need for vlan in this case
        // Clear out old group list. leave valid alone in case the port is disabled
        // for other reasons.
        entry->clear_grouplist();
       
    }
}

#endif


void
DcgmFabricParser::modifyRoutingTable( void )
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
               for ( j = 0; j < access.rmappolicytable_size(); j++ )
               {
                   modifyOneIngressRmapEntry( nodeId, physicalId,
                                              access.localportnum(),
                                              (rmapPolicyEntry *) &access.rmappolicytable(j));
               }

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
#endif
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

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)

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
#endif

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
DcgmFabricParser::pruneOneAccessPort ( uint32_t nodeId, uint32_t physicalId,
                                       lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        PRINT_ERROR("", "Invalid lwswitch.");
        return;
    }

    PRINT_INFO("%d %d %d",
               "prune lwswitch nodeId %d, physicalId %d, localportNum %d\n",
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
        PRINT_INFO("%d %d %d", "remove port info %d/%d/%d",
                    key.nodeId, key.physicalId, key.portIndex);
        portInfo.erase( it );
    }
}

/*
 * prune all access ports that are directly connected to a pruned GPU
 */
void
DcgmFabricParser::pruneAccessPorts( void )
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

           for ( i = 0; i < NUM_PORTS_PER_LWSWITCH; i++ )
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
DcgmFabricParser::pruneOneGpu ( uint32_t gpuEndpointID )
{
    int i, nodeId = gpuEndpointID / MAX_NUM_GPUS_PER_NODE;
    uint32_t physicalId = gpuEndpointID % MAX_NUM_GPUS_PER_NODE;

    if ( nodeId > mpFabric->fabricnode_size() )
    {
        PRINT_ERROR("%d", "invalid node %d.", nodeId);
        return;
    }

    PRINT_INFO("%d %d", "prune GPU nodeId %d, gpuEndpointID %d.",
                nodeId, gpuEndpointID);


    node *pNode = (node *)&mpFabric->fabricnode(getNodeIndex(nodeId));
    for ( i = 0; i < pNode->gpu_size(); i++ )
    {
        const GPU &gpu = pNode->gpu(i);

        if ( gpu.has_physicalid() && ( gpu.physicalid()== physicalId ) )
        {
            break;
        }

        if ( gpu.has_fabricaddrbase() &&
             ( gpu.fabricaddrbase() == GPU_FABRIC_DEFAULT_ADDR_BASE(gpuEndpointID) ) )
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
        PRINT_INFO("%d %d", "Prune GPU %d/%d", nodeId, physicalId);
        gpuCfg.erase( it );
    }
}

void
DcgmFabricParser::pruneGpus( void )
{
    std::set<uint32_t>::iterator it;

    if ( mDisabledGpuEndpointIds.size() == 0 )
        return;

    for ( it = mDisabledGpuEndpointIds.begin(); it != mDisabledGpuEndpointIds.end(); it++ )
    {
        pruneOneGpu( *it );
    }
}

/*
 * add a GPU's endPointedId to mDisabledGpuEndpointIds
 */
void
DcgmFabricParser::disableGpu ( GpuKeyType &gpu )
{
    int i;
    uint32_t gpuEndpointID = gpu.nodeId * MAX_NUM_GPUS_PER_NODE + gpu.physicalId;
    uint32_t ingressReqIndex = gpuEndpointID << 2;
    uint32_t ingressRespIndex = gpuEndpointID * NUM_LWLINKS_PER_GPU;

    mDisabledGpuEndpointIds.insert(gpuEndpointID);

    for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++ )
    {
        // ilwalidate ingress request entries to this GPU
        mIlwalidIngressReqEntries.insert( i + ingressReqIndex );
    }

    for ( i = 0; i < NUM_LWLINKS_PER_GPU; i++ )
    {
        // ilwalidate ingress response entries to this GPU
        mIlwalidIngressRespEntries.insert( i + ingressRespIndex );
    }


#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)

    uint32_t ingressRmapIndex = gpuEndpointID * NUM_INGR_RMAP_ENTRIES_PER_AMPERE;
    for ( i = 0; i < NUM_INGR_RMAP_ENTRIES_PER_AMPERE; i++ )
    {
        // ilwalidate ingress request entries to this GPU
        // This is for GPA
        mIlwalidIngressRmapEntries.insert( i + ingressRmapIndex );

        // This is for FLA
        mIlwalidIngressRmapEntries.insert( LR_FIRST_FLA_RMAP_SLOT + i + ingressRmapIndex );
    }

    // GPA and FLA share endpoint IDs so only one RID and RLAN for those

    mIlwalidIngressRidEntries.insert( gpuEndpointID );

    mIlwalidIngressRlanEntries.insert( gpuEndpointID );

#endif

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
DcgmFabricParser::pruneOneTrunkPort ( uint32_t nodeId, uint32_t physicalId,
                                      lwSwitch *pLwswitch, uint32_t localportNum )
{
    int i;

    if ( !pLwswitch )
    {
        PRINT_ERROR("", "Invalid lwswitch.");
        return;
    }

    PRINT_INFO("%d %d %d",
               "prune lwswitch nodeId %d, physicalId %d, localportNum %d\n",
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
        PRINT_WARNING("%d %d %d",
                      "trunk port lwswitch nodeId %d, physicalId %d, localportNum %d not found.\n",
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
        PRINT_INFO("%d %d %d", "remove port info %d/%d/%d",
                    key.nodeId, key.physicalId, key.portIndex);
        portInfo.erase( it );
    }
}

/*
 * prune all trunk ports that are directly connected to a pruned Switch
 */
void
DcgmFabricParser::pruneTrunkPorts( void )
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

           for ( i = 0; i < NUM_PORTS_PER_LWSWITCH; i++ )
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
DcgmFabricParser::pruneOneSwitch ( SwitchKeyType &key )
{
    int i;

    if ( key.nodeId > (uint32)mpFabric->fabricnode_size() )
    {
        PRINT_ERROR("", "Invalid node.");
        return;
    }

    PRINT_INFO("%d %d", "Prune switch nodeId %d, physicalId %d.",
                key.nodeId, key.physicalId);

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
        PRINT_INFO("%d %d", "Prune Switch %d/%d", key.nodeId, key.physicalId);
        lwswitchCfg.erase( it );
    }
}

void
DcgmFabricParser::pruneSwitches( void )
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
DcgmFabricParser::disableSwitch ( SwitchKeyType &key )
{
    mDisabledSwitches.insert(key);
}

void
DcgmFabricParser::modifyFabric( PruneFabricType pruneType )
{
    if ( ( pruneType == PRUNE_ALL ) ||  ( pruneType == PRUNE_SWITCH ) )
    {
        pruneSwitches();
        pruneTrunkPorts();
    }

    if ( ( pruneType == PRUNE_ALL ) ||  ( pruneType == PRUNE_GPU ) )
    {
        pruneGpus();

        pruneAccessPorts();

        modifyRoutingTable();

    }

    ofstream  outFile;
    outFile.open( DEFAULT_RUNNING_TOPOLOGY_FILE, ios::binary );

    if ( outFile.is_open() == false )
    {
        PRINT_ERROR("%s %s", "Failed to open output file %s, error is %s.",
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
}

int
DcgmFabricParser::getNumDisabledGpus( void )
{
    return mDisabledGpuEndpointIds.size();
}

int
DcgmFabricParser::getNumDisabledSwitches( void )
{
    return mDisabledSwitches.size();
}

void
DcgmFabricParser::fabricParserCleanup( void )
{
    // TODO
}

bool
DcgmFabricParser::isSwtichGpioPresent( void )
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

// given nodeIndex, return nodeId
uint32_t
DcgmFabricParser::getNodeId(uint32_t nodeIndex)
{
    uint32_t nodeId = nodeIndex;

    if ( nodeIndex > (uint32_t)mpFabric->fabricnode_size() )
    {
        PRINT_ERROR("%d", "Invalid nodeIndex %d", nodeIndex);
        return nodeId;
    }

    const node &fnode = mpFabric->fabricnode(nodeIndex);
    if ( fnode.has_nodeid() )
        nodeId = fnode.nodeid();

    return nodeId;
}

// given nodeId, return nodeIndex
uint32_t
DcgmFabricParser::getNodeIndex(uint32_t nodeId)
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

accessPort *
DcgmFabricParser::getAccessPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    accessPort *port = NULL;

    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        {
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.has_physicalid() == physicalId ) )
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
        }
    }

    return port;
}

trunkPort *
DcgmFabricParser::getTrunkPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum )
{
    trunkPort *port = NULL;
    for (int n = 0; n < (int)mpFabric->fabricnode_size(); n++ )
    {
        const node &fnode = mpFabric->fabricnode(n);
        if ( fnode.has_nodeid() && ( fnode.nodeid() == nodeId ) )
        {
            for (int w = 0; w < (int)fnode.lwswitch_size(); w++ )
            {
                const lwSwitch &lwswitch = fnode.lwswitch(w);
                if ( lwswitch.has_physicalid() &&
                    ( lwswitch.has_physicalid() == physicalId ) )
                {
                    for (int p = 0; p < (int)lwswitch.trunk_size(); p++ )
                    {
                        const trunkPort &trunk = lwswitch.trunk(p);
                        if ( trunk.has_localportnum() &&
                             ( trunk.localportnum() == portNum ) )
                        {
                            port = (trunkPort *)&access;
                            return port;
                        }
                    }
                }
            }
        }
    }

    return port;
}

const char *
DcgmFabricParser::getPlatformId()
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

