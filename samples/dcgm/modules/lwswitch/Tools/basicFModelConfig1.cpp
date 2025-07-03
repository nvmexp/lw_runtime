#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "basicFModelConfig1.h"

basicFModelConfig1::basicFModelConfig1( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x1 << 36;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 4;

    gpuFabricAddrBase[1]  = (uint64_t)0x02 << 36;
    gpuFabricAddrRange[1] = FAB_ADDR_RANGE_16G * 4;
};

basicFModelConfig1::~basicFModelConfig1()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void basicFModelConfig1::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 0 
        for ( portIndex = 0; portIndex <= 0; portIndex++ )
        {

            gpu   = gpus[nodeIndex][1];
            index = gpu->gpabase() >> 36;
            range = gpu->gparange();

            makeOneRemapEntry( nodeIndex,           // nodeIndex
                               swIndex,             // swIndex
                               portIndex,           // portIndex
                               index,               // ingress req table index
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
        }
        //access port 1
        for ( portIndex = 1; portIndex <= 1; portIndex++ )
        {

            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;
            range = gpu->gparange();

            makeOneRemapEntry( nodeIndex,           // nodeIndex
                               swIndex,             // swIndex
                               portIndex,           // portIndex
                               index,               // ingress req table index
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
        }
    }
}

void basicFModelConfig1::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, index, egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS], portCount;
    GPU * gpu;
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 0 
        for ( portIndex = 0; portIndex <= 0; portIndex++ )
        {

            gpu   = gpus[nodeIndex][1];
            index = gpu->gpabase() >> 36;
            portCount = 1;
            vcMap[0] = 0;
            egressPort[0] = 1;

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
        //access port 1
        // Access port 0 
        for ( portIndex = 1; portIndex <= 1; portIndex++ )
        {

            gpu   = gpus[nodeIndex][0];
            index = gpu->gpabase() >> 36;
            portCount = 1;
            vcMap[0] = 0;
            egressPort[0] = 0;

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

void basicFModelConfig1::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, index;
    GPU * gpu;
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 0 
        for ( portIndex = 0; portIndex <= 0; portIndex++ )
        {

            gpu   = gpus[nodeIndex][1];
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
        //access port 1
        // Access port 0 
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
        
    }

}

void basicFModelConfig1::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void basicFModelConfig1::makeAccessPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        //                nodeIndex swIndex     portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          0,        0,        0,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          1,        0,        1,        1,         DC_COUPLED);
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void basicFModelConfig1::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        //               nodeIndex swIndex portIndex farNodeID farSwitchID farPortNum
        // No trunk ports
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void basicFModelConfig1::makeOneLwswitch( int nodeIndex, int swIndex )
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
        PRINT_ERROR("%d,%d", "Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void basicFModelConfig1::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
{
    int i, j;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, ((nodeIndex * 8) + i), 0, 0x3F, 0xFFFFFFFF,
                    gpuFabricAddrBase[i], gpuFabricAddrRange[i], i);

    }

    // Add LimeRocks
    for ( i = 0; i < lrNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }
}

void basicFModelConfig1::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    for (nodeIndex = 0; nodeIndex < BASIC_FMODEL1_NUM_NODES; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        // set up node IP address
        //nodeip << "192.168.0." << (nodeIndex + 1);
        //nodes[nodeIndex]->set_ipaddress( nodeip.str().c_str() );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 2 GPUs, 1 LimeRock
            makeOneNode( nodeIndex, 2, 1);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void basicFModelConfig1::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void basicFModelConfig1::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicFModelConfig1::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
