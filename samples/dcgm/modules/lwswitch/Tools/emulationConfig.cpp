#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "emulationConfig.h"

emulationConfig::emulationConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x6AC << 34;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 2;
};

emulationConfig::~emulationConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void emulationConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr = 0, range = 0;
    GPU *gpu;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 16 to 16
        for ( portIndex = 16; portIndex <= 16; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < ( range/FAB_ADDR_RANGE_16G ); i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index & 0xFFFB ) << 34; // clear bit 36 for GPU
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00000000,    // vcModeValid15_8, port 8 to 13
                                        0x00000001,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    }
}

void emulationConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, index, enpointID, outPortNum;
    accessPort *outPort;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 16; portIndex <= 16; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort 16
            for ( outPortNum = 16; outPortNum <= 16; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000001,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }
        }
    }
}

void emulationConfig::makeGangedLinkTable( int nodeIndex, int willowIndex )
{
    return;
}

void emulationConfig::makeAccessPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          16,        0,        0,       16,        AC_COUPLED);
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void emulationConfig::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //               nodeIndex willowIndex portIndex farNodeID farSwitchID farPortNum
        // No trunk ports
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void emulationConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        switches[nodeIndex][willowIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][willowIndex]->set_ecid( ecid.str().c_str() );

        // Configure access ports
        makeAccessPorts( nodeIndex, willowIndex );

        // Configure trunk ports
        makeTrunkPorts( nodeIndex, willowIndex );

        // Configure ingress request table
        makeIngressReqTable( nodeIndex, willowIndex );

        // Configure egress request table
        makeIngressRespTable( nodeIndex, willowIndex );
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void emulationConfig::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
{
    int i, j;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, ((nodeIndex * 8) + i), 0, 0x3F, 0xFFFFFFFF,
                    gpuFabricAddrBase[i], gpuFabricAddrRange[i], i);
    }

    // Add Willows
    for ( i = 0; i < willowNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneWillow( nodeIndex, i);
    }
}

void emulationConfig::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    for (nodeIndex = 0; nodeIndex < EMULATION_NUM_NODES; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 1 GPUs, 1 Willow
            makeOneNode( nodeIndex, 1, 1);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
emulationConfig::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
emulationConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
emulationConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
emulationConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
