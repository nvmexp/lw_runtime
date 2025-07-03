#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include "json.h"

#include "fabricConfig.h"
#include "fm_log.h"
#include "FMCommonTypes.h"

// generate the topology names from topology enum
#define MAKE_STRINGS(VAR) #VAR,
const char* const fabricTopologyName[] = {
    FABRIC_TOPOLOGY_ENUM(MAKE_STRINGS)
};

fabricConfig::fabricConfig( fabricTopologyEnum topology )
{
    mFabric        = new( fabric );
    fabricTopology = topology;
    mSystemPartInfo = NULL;

    mFabric->set_version( FABRIC_MANAGER_VERSION );

    setFabricTopologyName();
    setFabricTopologyTime();

    memset( nodes,         0, sizeof(nodes)         );
    memset( switches,      0, sizeof(switches)      );
    memset( gpus,          0, sizeof(gpus)          );
    memset( accesses,      0, sizeof(accesses)      );
    memset( trunks,        0, sizeof(trunks)        );
    memset( reqEntry,      0, sizeof(reqEntry)      );
    memset( respEntry,     0, sizeof(respEntry)     );
    memset( gangedLinkEntry,0,sizeof(gangedLinkEntry));
    memset( portConfigs,   0, sizeof(portConfigs)   );
    memset( nodes,         0, sizeof(nodes)         );
    memset( rmapEntry,     0, sizeof(rmapEntry)     );
    memset( rmapExtAEntry, 0, sizeof(rmapExtAEntry) );
    memset( rmapExtBEntry, 0, sizeof(rmapExtBEntry) );
    memset( ridEntry,      0, sizeof(ridEntry)      );
    memset( rlanEntry,     0, sizeof(rlanEntry)     );

    memset( gpuFabricAddrBase,  0, sizeof(gpuFabricAddrBase) );
    memset( gpuFabricAddrRange, 0, sizeof(gpuFabricAddrRange));
    memset( gpuGpaEgmAddrBase,  0, sizeof(gpuGpaEgmAddrBase) );
    memset( gpuGpaEgmAddrRange, 0, sizeof(gpuGpaEgmAddrRange));
    memset( gpuFlaAddrBase,     0, sizeof(gpuFlaAddrBase) );
    memset( gpuFlaAddrRange,    0, sizeof(gpuFlaAddrRange));
}

fabricConfig::~fabricConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
}

/*
 * Expecting gpuPhysicalId to be contiguous, as it is used as index to gpus array
 * If gpuPhysicalId is not contiguous as switchPhysicalId, a mapping from
 * gpuPhysicalId to gpuIndex would be needed.
 */
void fabricConfig::makeOneGpu( int nodeIndex,     int gpuIndex, int endpointID,  int peerID,
                               int peerIdPortMap, int logicalToPhyPortMap,
                               int64_t fabricAddr,int64_t fabricAddrRange, int gpuPhysicalID)
{
    int i;
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_G" << gpuIndex;
    gpus[nodeIndex][gpuIndex]->set_physicalid( gpuPhysicalID );

    gpus[nodeIndex][gpuIndex]->set_version( FABRIC_MANAGER_VERSION );
    gpus[nodeIndex][gpuIndex]->set_ecid( ecid.str().c_str() );

    if (fabricAddr != 0 )
    {
        gpus[nodeIndex][gpuIndex]->set_fabricaddrbase( fabricAddr );
        gpus[nodeIndex][gpuIndex]->set_gpabase( fabricAddr );
    } else
    {
        gpus[nodeIndex][gpuIndex]->set_fabricaddrbase( GPU_FABRIC_DEFAULT_ADDR_BASE(endpointID) );
        gpus[nodeIndex][gpuIndex]->set_gpabase( GPU_FABRIC_DEFAULT_ADDR_BASE(endpointID) );
    }

    if ( fabricAddrRange != 0 )
    {
        gpus[nodeIndex][gpuIndex]->set_fabricaddrrange( fabricAddrRange );
        gpus[nodeIndex][gpuIndex]->set_gparange( fabricAddrRange );
    } else
    {
        gpus[nodeIndex][gpuIndex]->set_fabricaddrrange( GPU_FABRIC_DEFAULT_ADDR_RANGE );
        gpus[nodeIndex][gpuIndex]->set_gparange( GPU_FABRIC_DEFAULT_ADDR_RANGE );
    }

#if 0  // TODO need to change to the new peerIDMap format
    // for each GPU peer ID, specify all outputs are available
    // Add peerIDPortMap
    for ( i = 0; i < NUM_LWLINKS_PER_GPU; i++ )
    {
        if ( ( peerID != 0 ) && ( peerIdPortMap != 0X3F ) )
        {
            //  TODO: still need to verify with RM,
            //        RM programs default peerID 0 to represent the fabric
            //            program default peerIdPortMap to 0x3F to use all 6 lwlinks
            gpuPeerIdMaps[nodeIndex][gpuIndex][i] = gpus[nodeIndex][gpuIndex]->add_peertoport();
            gpuPeerIdMaps[nodeIndex][gpuIndex][i]->set_version( FABRIC_MANAGER_VERSION );
            gpuPeerIdMaps[nodeIndex][gpuIndex][i]->set_peerid( peerID );
            gpuPeerIdMaps[nodeIndex][gpuIndex][i]->set_portmap( peerIdPortMap );
        }
    }

    // configure GPU logical to physical port map
    if ( (unsigned int)logicalToPhyPortMap != 0xFFFFFFFF )
    {
        // TODO: set logical to physical port mapping
        //       default is 0xFFFFFFFF, not logical to physical port mapping
        gpus[nodeIndex][gpuIndex]->set_logicaltophyportmap( logicalToPhyPortMap );
    }
#endif
}

void fabricConfig::makeOneGpu( int nodeIndex,     int gpuIndex,  int endpointID,  int peerID,
                               int peerIdPortMap, int logicalToPhyPortMap,
                               int64_t gpaFabricAddr,int64_t gpaFabricAddrRange,
                               int64_t flaFabricAddr,int64_t flaFabricAddrRange,
                               int gpuPhysicalID)
{
    makeOneGpu( nodeIndex, gpuIndex, endpointID, peerID, peerIdPortMap, logicalToPhyPortMap,
                gpaFabricAddr, gpaFabricAddrRange, gpuPhysicalID );

    if (flaFabricAddr != 0 )
    {
        gpus[nodeIndex][gpuIndex]->set_flabase( flaFabricAddr );
    } else
    {
        gpus[nodeIndex][gpuIndex]->set_flabase( GPU_FABRIC_DEFAULT_ADDR_BASE(endpointID) );
    }

    if ( flaFabricAddrRange != 0 )
    {
        gpus[nodeIndex][gpuIndex]->set_flarange( flaFabricAddrRange );
    } else
    {
        gpus[nodeIndex][gpuIndex]->set_flarange( GPU_FABRIC_DEFAULT_ADDR_RANGE );
    }
}

void fabricConfig::makeOneGpu( int nodeIndex,     int gpuIndex,  int endpointID,  int peerID,
                               int peerIdPortMap, int logicalToPhyPortMap,
                               int64_t gpaFabricAddr,int64_t gpaFabricAddrRange,
                               int64_t flaFabricAddr,int64_t flaFabricAddrRange,
                               int gpuPhysicalID, uint32_t targetId )
{
    makeOneGpu( nodeIndex, gpuIndex, endpointID, peerID, peerIdPortMap,logicalToPhyPortMap,
                gpaFabricAddr,gpaFabricAddrRange, flaFabricAddr, flaFabricAddrRange, gpuPhysicalID);
    gpus[nodeIndex][gpuIndex]->set_targetid( targetId );
}

void fabricConfig::makeOneGpu( int nodeIndex, int gpuPhysicalID, uint32_t targetId )
{
    int i;
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_G" << gpuPhysicalID;
    gpus[nodeIndex][gpuPhysicalID]->set_physicalid( gpuPhysicalID );
    gpus[nodeIndex][gpuPhysicalID]->set_targetid( targetId );


    gpus[nodeIndex][gpuPhysicalID]->set_version( FABRIC_MANAGER_VERSION );
    gpus[nodeIndex][gpuPhysicalID]->set_ecid( ecid.str().c_str() );

    gpus[nodeIndex][gpuPhysicalID]->set_fabricaddrbase( gpuFabricAddrBase[targetId] );
    gpus[nodeIndex][gpuPhysicalID]->set_fabricaddrrange( gpuFabricAddrRange[targetId] );

    gpus[nodeIndex][gpuPhysicalID]->set_gpabase( gpuFabricAddrBase[targetId] );
    gpus[nodeIndex][gpuPhysicalID]->set_gparange( gpuFabricAddrRange[targetId] );

    gpus[nodeIndex][gpuPhysicalID]->set_flabase( gpuFlaAddrBase[targetId] );
    gpus[nodeIndex][gpuPhysicalID]->set_flarange( gpuFlaAddrRange[targetId] );

    gpus[nodeIndex][gpuPhysicalID]->set_gpaegmbase( gpuGpaEgmAddrBase[targetId] );
    gpus[nodeIndex][gpuPhysicalID]->set_gpaegmrange( gpuGpaEgmAddrRange[targetId] );
}

void fabricConfig::makeOneIngressReqEntry( int nodeIndex, int willowIndex, int portIndex,
                                           int index, int64_t address, int routePolicy,
                                           int vcModeValid7_0, int vcModeValid15_8,
                                           int vcModeValid17_16, int entryValid)
{
    ingressRequestTable * entry = NULL;

    if ( index >= INGRESS_REQ_TABLE_SIZE )
    {
        FM_LOG_ERROR( "Invalid Ingress Req Index %d.\n", index);
        return;
    }

    if ( accesses[nodeIndex][willowIndex][portIndex] != NULL )
    {
        entry = accesses[nodeIndex][willowIndex][portIndex]->add_reqrte();
    }
    else if ( trunks[nodeIndex][willowIndex][portIndex] != NULL )
    {
        entry = trunks[nodeIndex][willowIndex][portIndex]->add_reqrte();
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d willowIndex %d, portIndex %d\n",
                    nodeIndex, willowIndex, portIndex);
    }

    reqEntry[nodeIndex][willowIndex][portIndex][index] = entry;
    entry->set_version( FABRIC_MANAGER_VERSION );
    entry->set_index( index );
    entry->set_address( address );
    entry->set_routepolicy( routePolicy );
    entry->set_vcmodevalid7_0( vcModeValid7_0 );
    entry->set_vcmodevalid15_8( vcModeValid15_8 );
    entry->set_vcmodevalid17_16( vcModeValid17_16 );
    entry->set_entryvalid( entryValid );
}

void fabricConfig::makeOneIngressRespEntry( int nodeIndex, int willowIndex, int portIndex,
                                            int index, int routePolicy,
                                            int vcModeValid7_0, int vcModeValid15_8,
                                            int vcModeValid17_16, int entryValid )
{

    ingressResponseTable * entry = NULL;
    if ( index >= INGRESS_RESP_TABLE_SIZE )
    {
        FM_LOG_ERROR( "Invalid Ingress Resp Index %d.\n", index);
        return;
    }

    if ( accesses[nodeIndex][willowIndex][portIndex] != NULL )
    {
        entry = accesses[nodeIndex][willowIndex][portIndex]->add_rsprte();
    }
    else if ( trunks[nodeIndex][willowIndex][portIndex] != NULL )
    {
        entry = trunks[nodeIndex][willowIndex][portIndex]->add_rsprte();
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d willonwIndex %d, portIndex %d\n",
                    nodeIndex, willowIndex, portIndex);
    }

    respEntry[nodeIndex][willowIndex][portIndex][index] = entry;
    entry->set_version( FABRIC_MANAGER_VERSION );
    entry->set_index( index );
    entry->set_routepolicy( routePolicy );
    entry->set_vcmodevalid7_0( vcModeValid7_0 );
    entry->set_vcmodevalid15_8( vcModeValid15_8 );
    entry->set_vcmodevalid17_16( vcModeValid17_16 );
    entry->set_entryvalid( entryValid );
}

void fabricConfig::makeOneGangedLinkEntry( int nodeIndex, int willowIndex, int portIndex,
                                           int index, int data )
{
    int32_t *entry;
    gangedLinkTable *table = NULL;

    if ( accesses[nodeIndex][willowIndex][portIndex] != NULL )
    {
        if ( accesses[nodeIndex][willowIndex][portIndex]->has_gangedlinktbl() )
        {
            table = (gangedLinkTable *)&accesses[nodeIndex][willowIndex][portIndex]->gangedlinktbl();
        }
        else
        {
            table = new gangedLinkTable;
            accesses[nodeIndex][willowIndex][portIndex]->set_allocated_gangedlinktbl(table);
        }
    }
    else if ( trunks[nodeIndex][willowIndex][portIndex] != NULL )
    {
        if ( trunks[nodeIndex][willowIndex][portIndex]->has_gangedlinktbl() )
        {
            table = (gangedLinkTable *)&trunks[nodeIndex][willowIndex][portIndex]->gangedlinktbl();
        }
        else
        {
            table = new gangedLinkTable;
            trunks[nodeIndex][willowIndex][portIndex]->set_allocated_gangedlinktbl(table);
        }
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d willonwIndex %d, portIndex %d\n",
                    nodeIndex, willowIndex, portIndex);
    }

    entry = new int32_t;
    *entry = data;
    gangedLinkEntry[nodeIndex][willowIndex][portIndex][index] = entry;
    table->set_version( FABRIC_MANAGER_VERSION );
    table->add_data( data );
}

void fabricConfig::makeOneRemapEntry( int nodeIndex,     int swIndex,         int portIndex,
                                      int index,         int entryValid,      int64_t address,
                                      int reqContextChk, int reqContextMask,  int reqContextRep,
                                      int addressOffset, int addressBase,     int addressLimit,
                                      int targetId,      int rmapFlags,       int irlSelect,
                                      int p2rSwizEnable, int mult2,           int planeSelect,
                                      RemapTable remapTable )
{
    rmapPolicyEntry * entry = NULL;

    if ( index >= INGRESS_REMAP_TABLE_SIZE )
    {
        FM_LOG_ERROR("Invalid Ingress Req Index %d.\n", index);
        return;
    }

    if ( trunks[nodeIndex][swIndex][portIndex] != NULL )
    {
        FM_LOG_ERROR("Attempt to set remap entry on trunk port, nodeIndex %d swIndex %d, portIndex %d\n",
                    nodeIndex, swIndex, portIndex);
        return;
    }
    if ( accesses[nodeIndex][swIndex][portIndex] != NULL )
    {
        switch (remapTable) {
        case NORMAL_RANGE:
            entry = accesses[nodeIndex][swIndex][portIndex]->add_rmappolicytable();
            rmapEntry[nodeIndex][swIndex][portIndex][index] = entry;
            break;

        case EXTENDED_RANGE_A:
            entry = accesses[nodeIndex][swIndex][portIndex]->add_extarmappolicytable();
            rmapExtAEntry[nodeIndex][swIndex][portIndex][index] = entry;
            break;

        case EXTENDED_RANGE_B:
            entry = accesses[nodeIndex][swIndex][portIndex]->add_extbrmappolicytable();
            rmapExtBEntry[nodeIndex][swIndex][portIndex][index] = entry;
            break;

        default:
            entry = accesses[nodeIndex][swIndex][portIndex]->add_rmappolicytable();
            rmapEntry[nodeIndex][swIndex][portIndex][index] = entry;
            break;
        }
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d swIndex %d, portIndex %d\n",
                    nodeIndex, swIndex, portIndex);
    }

    entry->set_version( FABRIC_MANAGER_VERSION );
    entry->set_index( index );
    entry->set_entryvalid( entryValid );
    entry->set_address( address );
    entry->set_reqcontextchk( reqContextChk );
    entry->set_reqcontextmask( reqContextMask );
    entry->set_reqcontextrep( reqContextRep );
    entry->set_addressoffset( addressOffset );
    entry->set_addressbase( addressBase );
    entry->set_addresslimit( addressLimit );
    entry->set_targetid( targetId );
    entry->set_remapflags( rmapFlags );
    entry->set_irlselect( irlSelect );
    entry->set_p2rswizenable( p2rSwizEnable );
    entry->set_mult2( mult2 );
    entry->set_planeselect( planeSelect );
}

void fabricConfig::makeOneRIDRouteEntry( int nodeIndex, int swIndex,     int portIndex,
                                         int index,     int entryValid,  int rMod,
                                         int portCount, int *vcMap,      int *egressPort)
{
    int i;
    ridRouteEntry * entry = NULL;
    routePortList * egress[INGRESS_RID_MAX_PORTS] = { NULL };

    if ( index > INGRESS_RID_TABLE_SIZE )
    {
        FM_LOG_ERROR("Invalid Ingress Resp Index %d.\n", index);
        return;
    }

    if ( portCount > INGRESS_RID_MAX_PORTS )
    {
        FM_LOG_ERROR("Invalid Ingress Port Count %d.\n", portCount);
        return;
    }

    if ( accesses[nodeIndex][swIndex][portIndex] != NULL )
    {
        entry = accesses[nodeIndex][swIndex][portIndex]->add_ridroutetable();
    }
    else if ( trunks[nodeIndex][swIndex][portIndex] != NULL )
    {
        entry = trunks[nodeIndex][swIndex][portIndex]->add_ridroutetable();
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d willonwIndex %d, portIndex %d\n",
                    nodeIndex, swIndex, portIndex);
    }

    ridEntry[nodeIndex][swIndex][portIndex][index] = entry;
    entry->set_version( FABRIC_MANAGER_VERSION );
    entry->set_index( index );
    entry->set_valid( entryValid );
    entry->set_rmod( rMod );

    for ( i = 0; i < portCount; i++ )
    {
        egress[i] = entry->add_portlist();
        egress[i]->set_vcmap( vcMap[i] );
        egress[i]->set_portindex( egressPort[i] );
    }
}

void fabricConfig::makeOneRLANRouteEntry( int nodeIndex,    int swIndex, int portIndex,
                                          int index,        int entryValid,  int groupCount,
                                          int *groupSelect, int *groupSize)
{
    int i;
    rlanRouteEntry * entry = NULL;
    rlanGroupSel * group[INGRESS_RLAN_MAX_GROUPS] = { NULL };

    if ( index >= INGRESS_RLAN_TABLE_SIZE )
    {
        FM_LOG_ERROR("Invalid Ingress Resp Index %d.\n", index);
        return;
    }

    if ( accesses[nodeIndex][swIndex][portIndex] != NULL )
    {
        entry = accesses[nodeIndex][swIndex][portIndex]->add_rlanroutetable();
    }
    else if ( trunks[nodeIndex][swIndex][portIndex] != NULL )
    {
        entry = trunks[nodeIndex][swIndex][portIndex]->add_rlanroutetable();
    }
    else
    {
        FM_LOG_ERROR("Invalid port nodeIndex %d willonwIndex %d, portIndex %d\n",
                    nodeIndex, swIndex, portIndex);
    }

    rlanEntry[nodeIndex][swIndex][portIndex][index] = entry;
    entry->set_version( FABRIC_MANAGER_VERSION );
    entry->set_index( index );
    entry->set_valid( entryValid );


    for ( i = 0; i < groupCount; i++ )
    {

        group[i] = entry->add_grouplist();
        group[i]->set_groupselect( groupSelect[i] );
        group[i]->set_groupsize( groupSize[i] );
    }
}

// $$$TODO add a parameter for accessConnect type to distinguish between GPU and CPU connections
// Hard coding to GPU for now

void fabricConfig::makeOneAccessPort( int nodeIndex, int willowIndex, int portIndex,
                                      int farNodeID, int farPeerID,   int farPortNum,
                                      PhyMode phyMode)
{
    accesses[nodeIndex][willowIndex][portIndex] = switches[nodeIndex][willowIndex]->add_access();
    accesses[nodeIndex][willowIndex][portIndex]->set_version( FABRIC_MANAGER_VERSION );
    accesses[nodeIndex][willowIndex][portIndex]->set_localportnum(portIndex);
    accesses[nodeIndex][willowIndex][portIndex]->set_farnodeid(farNodeID);
    accesses[nodeIndex][willowIndex][portIndex]->set_farpeerid(farPeerID);
    accesses[nodeIndex][willowIndex][portIndex]->set_farportnum(farPortNum);
    accesses[nodeIndex][willowIndex][portIndex]->set_connecttype( ACCESS_GPU_CONNECT );

    portConfigs[nodeIndex][willowIndex][portIndex] = new switchPortConfig();
    portConfigs[nodeIndex][willowIndex][portIndex]->set_version( FABRIC_MANAGER_VERSION );
    portConfigs[nodeIndex][willowIndex][portIndex]->set_type( ACCESS_PORT_GPU );
    portConfigs[nodeIndex][willowIndex][portIndex]->set_requesterlinkid( farPeerID * 6 + farPortNum );
    portConfigs[nodeIndex][willowIndex][portIndex]->set_phymode(phyMode);
    accesses[nodeIndex][willowIndex][portIndex]->set_allocated_config(portConfigs[nodeIndex][willowIndex][portIndex] );
}

void fabricConfig::makeOneAccessPort( int nodeIndex, int willowIndex, int portIndex,
                                      int farNodeID, int farPeerID,   int farPortNum,
                                      PhyMode phyMode, uint32_t farPeerTargetID, uint32_t rlanID,
                                      uint32_t maxTargetID )
{
    makeOneAccessPort(nodeIndex, willowIndex, portIndex, farNodeID, farPeerID, farPortNum, phyMode);
    switchPortConfig *portConfig = (switchPortConfig *)&accesses[nodeIndex][willowIndex][portIndex]->config();

    if (portConfig)
    {
        // On Limerock requesterLinkId needs to the same as the targetID of the connected GPU.
        portConfig->set_requesterlinkid( farPeerTargetID );

        portConfig->set_rlanid( rlanID );

        if ( maxTargetID > 0 )
        {
            portConfig->set_maxtargetid( (lwswitchMaxTargetId)maxTargetID );
        }
    }
}

// $$$TODO add a parameter for trunkConnect type
// $$$TODO add a parameter for sticky partition. Encoding in high bits of portIndex for now. 
// Hard coding to SWITCH for now

void fabricConfig::makeOneTrunkPort( int nodeIndex, int willowIndex, int portIndex,
                                     int farNodeID, int farSwitchID, int farPortNum,
                                     PhyMode phyMode, uint32_t maxTargetID )
{
    int enableVCSet1;
    enableVCSet1 = ( portIndex & 0x000F0000 ) >> 16;
    portIndex &= 0x0000FFFF;
    trunks[nodeIndex][willowIndex][portIndex] = switches[nodeIndex][willowIndex]->add_trunk();
    trunks[nodeIndex][willowIndex][portIndex]->set_version( FABRIC_MANAGER_VERSION );
    trunks[nodeIndex][willowIndex][portIndex]->set_localportnum(portIndex);
    trunks[nodeIndex][willowIndex][portIndex]->set_farnodeid(farNodeID);
    trunks[nodeIndex][willowIndex][portIndex]->set_farswitchid(farSwitchID);
    trunks[nodeIndex][willowIndex][portIndex]->set_farportnum(farPortNum);
    trunks[nodeIndex][willowIndex][portIndex]->set_connecttype( TRUNK_SWITCH_CONNECT );

    portConfigs[nodeIndex][willowIndex][portIndex] = new switchPortConfig();
    portConfigs[nodeIndex][willowIndex][portIndex]->set_version( FABRIC_MANAGER_VERSION );
    portConfigs[nodeIndex][willowIndex][portIndex]->set_type( TRUNK_PORT_SWITCH );
    portConfigs[nodeIndex][willowIndex][portIndex]->set_phymode(phyMode);

    if ( maxTargetID > 0 )
    {
        portConfigs[nodeIndex][willowIndex][portIndex]->set_maxtargetid( (lwswitchMaxTargetId)maxTargetID );
    }

    if ( enableVCSet1 != 0 ) 
    {
        portConfigs[nodeIndex][willowIndex][portIndex]->set_enablevcset1(1);
    }        
    trunks[nodeIndex][willowIndex][portIndex]->set_allocated_config(portConfigs[nodeIndex][willowIndex][portIndex] );
}

const char *fabricConfig::getFabricTopologyName( fabricTopologyEnum topology )
{
    if ( topology >= MAX_TOPOLOGY_CONFIG ) {
        return fabricTopologyName[UNKNOWN_TOPOLOGY];
    } else {
        return fabricTopologyName[topology];
    }
}

void fabricConfig::setFabricTopologyName()
{
    // set topology name
    mFabric->set_name(getFabricTopologyName(fabricTopology));
}

void fabricConfig::setFabricTopologyTime()
{
    // set topology build time
    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mFabric->set_time(asctime(loc_time));
}

void fabricConfig::makeOneLwswitch( int nodeIndex, int swIndex )
{
    printf("%s needs to be overridden by derived class.\n", __FUNCTION__);
    assert(0);
}

void fabricConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    printf("%s needs to be overridden by derived class.\n", __FUNCTION__);
    assert(0);
}

void fabricConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    printf("%s needs to be overridden by derived class.\n", __FUNCTION__);
    assert(0);
}

void fabricConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    printf("%s needs to be overridden by derived class.\n", __FUNCTION__);
    assert(0);
}


/*
 * The following is an example of specifying shared fabric partitions
 * in JSON format.
{
    "name": "customer shared fabric partition name",
    "time": "Fri Oct 18 11:11:11 2019",
    "partitionInfo": [
      {
        "partitionId": 8,
        "gpuPhysicalIds": [1, 2, 3, 4],
        "switchPhysicalIds": [8, 9, 10, 11, 12, 13]
      },
      {
        "partitionId": 9,
        "gpuPhysicalIds": [9, 10, 11, 12],
        "switchPhysicalIds": [24, 25, 26, 27, 28, 29]
      }
    ]
}
 */

bool
fabricConfig::switchOnBaseboard1( uint32_t physicalId )
{
    if ( ( physicalId == 0x08 ) || ( physicalId == 0x09 ) ||
         ( physicalId == 0x0A ) || ( physicalId == 0x0B ) ||
         ( physicalId == 0x0C ) || ( physicalId == 0x0D ) )
    {
        return true;
    }
    return false;
}

bool
fabricConfig::switchOnBaseboard2( uint32_t physicalId )
{
    if ( ( physicalId == 0x18 ) || ( physicalId == 0x19 ) ||
         ( physicalId == 0x1A ) || ( physicalId == 0x1B ) ||
         ( physicalId == 0x1C ) || ( physicalId == 0x1D ) )
    {
        return true;
    }
    return false;
}

void
fabricConfig::parsePartitionJsonFile( uint32_t lwLinkIntraTrunkConnCount )
{
    std::ifstream partitionFile(mSharedPartitionJsonFile);
    Json::Reader reader;
    Json::Value root;

    int numMaxGpuPartitions = 0;
    int num1GpuPartitions = 0;

    if ( partitionFile.fail() )
    {
        printf("Failed to read customer shared fabric partition file %s\n",
                mSharedPartitionJsonFile);
        return;
    }

    if ( !reader.parse(partitionFile, root) )
    {
        printf("Failed to parse customer shared fabric partition file %s\n",
                mSharedPartitionJsonFile);
        return;
    }

    // set the version
    mSharedPartitiolwersion = FABRIC_PARTITION_VERSION;

    // get name, if no name is specified, use the filename
    std::string name = root.get("name", mSharedPartitionJsonFile).asString();

    // get time
    // set topology build time, if no time is specified use lwrtime
    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mSharedPartitionTimeStamp = root.get("time", asctime(loc_time)).asString();

    // loop through partitionInfo
    const Json::Value partitionList = root["partitionInfo"];
    uint numPartitions = partitionList.size();
    if (numPartitions > MAX_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS)
    {
        printf("Number of partitions %d is more than the max number of partition supported %d.\n",
                numPartitions, MAX_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS);
        return;
    }

    mNumSharedPartitions = numPartitions;

    uint i = 0, j;
    Json::Value::iterator it;
    for ( it = partitionList.begin(); it != partitionList.end(); it++, i++ )
    {
        const Json::Value partitionInfo = *it;

        // get partitionId
        if (!partitionInfo.isMember("partitionId"))
        {
            printf("Invalid partitionId at index %d.\n", i);
            return;
        }

        uint32_t partitionId = partitionInfo.get("partitionId", 0).asUInt();

        if ( !partitionInfo.isMember("gpuPhysicalIds") )
        {
            printf("gpuPhysicalIds is not specified at partitionId %d.\n", partitionId);
            return;
        }

        if ( !partitionInfo.isMember("switchPhysicalIds") )
        {
            // There is no switch needed for 1 GPU partitions
            printf("switchPhysicalIds is not specified at partitionId %d.\n", partitionId);
        }

        uint numGpus = partitionInfo["gpuPhysicalIds"].size();
        if ( numGpus >  MAX_NUM_GPUS_PER_NODE )
        {
            printf("numGpus %d at partitionId %d is more than the max number of GPUs supported %d.\n",
                    numGpus, partitionId, MAX_NUM_GPUS_PER_NODE);
            return;
        }

        uint numSwitches = partitionInfo["switchPhysicalIds"].size();
        if ( numSwitches > MAX_NUM_LWSWITCH_PER_NODE )
        {
            printf("numSwitches %d at partitionId %d is more than the max number of LWSwitch supported %d.\n",
                    numSwitches, partitionId, MAX_NUM_LWSWITCH_PER_NODE);
            return;
        }

        SharedPartInfoTable_t *pPartitionInfo = &(mSharedVMPartInfo[i]);
        pPartitionInfo->partitionId = partitionId;
        pPartitionInfo->gpuCount = numGpus;
        pPartitionInfo->switchCount = numSwitches;

        if ( numGpus == MAX_NUM_GPUS_PER_NODE ) numMaxGpuPartitions++;
        if ( numGpus == 1 ) num1GpuPartitions++;

        Json::Value::iterator jit;

        // get gpuPhysicalIds
        j = 0;
        for ( jit = partitionInfo["gpuPhysicalIds"].begin();
              jit != partitionInfo["gpuPhysicalIds"].end();
              jit++, j++ )
        {
            const Json::Value gpuPhysicalIds = *jit;

            struct SharedPartGpuInfoTable_t *pGpuInfo = &(mSharedVMPartInfo[i].gpuInfo[j]);
            pGpuInfo->physicalId = gpuPhysicalIds.asInt();

            // customer facing GPU physicalId is 1 based.
            if ( pGpuInfo->physicalId < 1 || pGpuInfo->physicalId > MAX_NUM_GPUS_PER_NODE )
            {
                printf("Invalid GPU physicalId %d at partitionId %d.\n",
                        pGpuInfo->physicalId, partitionId);
                return;
            }

            // internal GPU physicalId is 0 based.
            pGpuInfo->physicalId -= 1;
            pGpuInfo->numEnabledLinks = MAX_LWLINKS_PER_GPU;
        }

        // get switchPhysicalIds
        bool switchOnBB1 = false;
        bool switchOnBB2 = false;

        j = 0;
        for ( jit = partitionInfo["switchPhysicalIds"].begin();
              jit != partitionInfo["switchPhysicalIds"].end();
              jit++, j++ )
        {
            const Json::Value switchPhysicalIds = *jit;

            SharedPartSwitchInfoTable_t *pSwitchInfo = &(mSharedVMPartInfo[i].switchInfo[j]);
            pSwitchInfo->physicalId = switchPhysicalIds.asInt();

            if ( ( switchOnBaseboard1( pSwitchInfo->physicalId ) == false ) &&
                 ( switchOnBaseboard2( pSwitchInfo->physicalId ) == false ) )
            {
                printf("Invalid LWSwitch physicalId %d at partitionId %d.\n",
                        pSwitchInfo->physicalId, partitionId);
                return;
            }

            switchOnBB1 |= switchOnBaseboard1( pSwitchInfo->physicalId );
            switchOnBB2 |= switchOnBaseboard2( pSwitchInfo->physicalId );
        }

        // callwlate LWSwitch numEnabledLinks
        if ( switchOnBB1 && switchOnBB2 )
        {
            if ( numGpus < 8 )
            {
                printf("%d number of GPUs cannot accross baseboards.\n", numGpus);
            }
            // partition span 2 baseboards, use all trunk links
            pPartitionInfo->lwLinkIntraTrunkConnCount = lwLinkIntraTrunkConnCount;
        }
        else
        {
            pPartitionInfo->lwLinkIntraTrunkConnCount = 0;
        }

        uint32_t accessConnCount;
        if ( ( numGpus == MAX_NUM_GPUS_PER_NODE ) || ( numGpus == MAX_NUM_GPUS_PER_NODE/2 ) )
        {
            accessConnCount = 8 * 2; // 8 GPU per switches, 2 access link per GPU
        }
        else if ( numGpus > 1 )
        {
            accessConnCount = numGpus * 2; // 8 GPU per switches, 2 access link per GPU
        }
        else
        {
            accessConnCount = 0; // 1 GPU partition, no need for any access link
        }

        for ( uint j = 0; j < numSwitches; j++ )
        {
            SharedPartSwitchInfoTable_t *pSwitchInfo = &(mSharedVMPartInfo[i].switchInfo[j]);
            pSwitchInfo->numEnabledLinks = pPartitionInfo->lwLinkIntraTrunkConnCount + accessConnCount;
        }
    }

    if ( numMaxGpuPartitions > 1 )
    {
        printf("More than 1 %d GPU partitions are specified.\n", MAX_NUM_GPUS_PER_NODE);
    }

    if ( num1GpuPartitions > MAX_NUM_GPUS_PER_NODE )
    {
        printf("More than %d 1 GPU partitions are specified.\n", MAX_NUM_GPUS_PER_NODE);
    }
}


