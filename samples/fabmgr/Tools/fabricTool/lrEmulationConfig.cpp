#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "fabricConfig.h"
#include "lrEmulationConfig.h"

lrEmulationConfig::lrEmulationConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x1 << 36;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 4;

    gpuFlaAddrBase[0]  = (uint64_t)(0x1 + LR_FIRST_FLA_RMAP_SLOT) << 36;
    gpuFlaAddrRange[0] = FAB_ADDR_RANGE_16G * 4;
};

lrEmulationConfig::~lrEmulationConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void lrEmulationConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 32
        for ( portIndex = 32; portIndex <= 32; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];

            // GPA map slot
            index = gpu->gpabase() >> 36;
            range = gpu->gparange();

            makeOneRemapEntry( nodeIndex,           // nodeIndex
                               swIndex,             // swIndex
                               portIndex,           // portIndex
                               index,               // rmap table index
                               1,                   // entry valid
                               0,                   // 64 bits remap fabric address (remap to 0 physical)
                               0,                   // context match
                               0,                   // context mask
                               0,                   // context replacement
                               0,                   // address offset
                               0,                   // address base
                               0,                   // address limit
                               index,               // target ID
                               1,                   // rfunc = remap address. skip context and range checks
                               0,                   // irl select
                               0,                   // FAM swizzle 
                               0,                   // FAM mult2
                               0);                  // FAM plane select

            // FLA map slot
            index = gpu->flabase() >> 36;
            range = gpu->flarange();

            makeOneRemapEntry( nodeIndex,           // nodeIndex
                               swIndex,             // swIndex
                               portIndex,           // portIndex
                               index,               // rmap table index
                               1,                   // entry valid
                               gpu->flabase(),      // 64 bits remap FLA to itsef, due to bug 2498189
                               0,                   // context match
                               0,                   // context mask
                               0,                   // context replacement
                               0,                   // address offset
                               0,                   // address base
                               0,                   // address limit
                               (index-LR_FIRST_FLA_RMAP_SLOT), // target ID
                               1,                   // remap FLA address. skip context and range checks
                               0,                   // irl select
                               0,                   // FAM swizzle
                               0,                   // FAM mult2
                               0);                  // FAM plane select
        }
    }
}

void lrEmulationConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, index, egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS], portCount;
    GPU * gpu;
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 32
        for ( portIndex = 32; portIndex <= 32; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;
            portCount = 1;
            vcMap[0] = 0;
            egressPort[0] = 0; // egress Trunk port 0

            makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                  swIndex,          // swIndex
                                  portIndex,        // portIndex
                                  index,            // RID Route table index
                                  1,                // entry valid
                                  0,                // rmod (no special routing)
                                  portCount,        // number of ports
                                  vcMap,            // pointer to array of VC controls   
                                  egressPort);      // pointer to array of ports
        }

        // Trunk port 1
        for ( portIndex = 1; portIndex <= 1; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;
            portCount = 1;
            vcMap[0] = 0;
            egressPort[0] = 2; // egress Trunk port 2

            makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                  swIndex,          // swIndex
                                  portIndex,        // portIndex
                                  index,            // RID Route table index
                                  1,                // entry valid
                                  0,                // rmod (no special routing)
                                  portCount,        // number of ports
                                  vcMap,            // pointer to array of VC controls
                                  egressPort);      // pointer to array of ports
        }
        
        // Trunk port 3
        for ( portIndex = 3; portIndex <= 3; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;
            portCount = 1;
            vcMap[0] = 0;
            egressPort[0] = 32; // egress Access port 32

            makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                  swIndex,          // swIndex
                                  portIndex,        // portIndex
                                  index,            // RID Route table index
                                  1,                // entry valid
                                  0,                // rmod (no special routing)
                                  portCount,        // number of ports
                                  vcMap,            // pointer to array of VC controls
                                  egressPort);      // pointer to array of ports
        }
    }
}

void lrEmulationConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, index;
    GPU * gpu;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 32
        for ( portIndex = 32; portIndex <= 32; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;

            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   index,            // RID Route table index
                                   1,                // entry valid
                                   0,                // groupcount
                                   NULL,             // group select array
                                   NULL );           // group size array   
        }

        // Trunk port 1
        for ( portIndex = 1; portIndex <= 1; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;

            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   index,            // RID Route table index
                                   1,                // entry valid
                                   0,                // groupcount
                                   NULL,             // group select array
                                   NULL );           // group size array
        }

        // Trunk port 3
        for ( portIndex = 3; portIndex <= 3; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;

            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   index,            // RID Route table index
                                   1,                // entry valid
                                   0,                // groupcount
                                   NULL,             // group select array
                                   NULL );           // group size array
        }
    }
}

void lrEmulationConfig::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void lrEmulationConfig::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        farPeerID = 0;
        farPeerTargetID  = gpus[nodeIndex][farPeerID]->gpabase() >> 36;

        //                nodeIndex swIndex     portIndex farNodeID farPeerID farPortNum portMode    farPeerTargetID  rlanID
        makeOneAccessPort(0,        0,          32,       0,        farPeerID,  0,       DC_COUPLED, farPeerTargetID, 0);
    }
    else
    {
        printf("Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void lrEmulationConfig::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        //               nodeIndex swIndex portIndex farNodeID farSwitchID farPortNum portMode
        makeOneTrunkPort(0,        0,      0,        0,        0,          1,         DC_COUPLED);
        makeOneTrunkPort(0,        0,      1,        0,        0,          0,         DC_COUPLED);
        makeOneTrunkPort(0,        0,      2,        0,        0,          3,         DC_COUPLED);
        makeOneTrunkPort(0,        0,      3,        0,        0,          2,         DC_COUPLED);
    }
    else
    {
        printf("Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void lrEmulationConfig::makeOneLwswitch( int nodeIndex, int swIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << swIndex;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        switches[nodeIndex][swIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][swIndex]->set_ecid( ecid.str().c_str() );

        // Configure access ports
        makeAccessPorts( nodeIndex, swIndex );

        // Configure trunk ports
        makeTrunkPorts( nodeIndex, swIndex );

        // Configure ingress remap table
        makeRemapTable( nodeIndex, swIndex );

        // Configure ingress RID Route table
        makeRIDRouteTable( nodeIndex, swIndex );

        // Configure ingress RLAN Route table
        makeRLANRouteTable( nodeIndex, swIndex );

    }
    else
    {
        printf("Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void lrEmulationConfig::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
{
    int i, j;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, ((nodeIndex * 8) + i), 0, 0x3F, 0xFFFFFFFF,
                    gpuFabricAddrBase[i], gpuFabricAddrRange[i],
                    gpuFlaAddrBase[i], gpuFlaAddrRange[i],
                    i);

    }

    // Add LimeRocks
    for ( i = 0; i < lrNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }
}

void lrEmulationConfig::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        // set up node IP address
        //nodeip << "192.168.0." << (nodeIndex + 1);
        //nodes[nodeIndex]->set_ipaddress( nodeip.str().c_str() );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 1 GPUs, 1 LimeRock
            makeOneNode( nodeIndex, 1, 1);
            break;

        default:
            printf("Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void lrEmulationConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void lrEmulationConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void lrEmulationConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
