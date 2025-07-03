#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "basicE3600Config4.h"

basicE3600Config4::basicE3600Config4( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x6AC << 34;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 4;
};

basicE3600Config4::~basicE3600Config4()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void basicE3600Config4::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 16 to 17
        for ( portIndex = 16; portIndex <= 17; portIndex++ )
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
                else
                {
                    mappedAddr = ( (int64_t)index ) << 34; // clear bit 36 for GPU
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00000000,    // vcModeValid15_8, port 8 to 13
                                        0x00000011,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    }
}

void basicE3600Config4::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, index, enpointID, outPortNum;
    accessPort *outPort;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 16 to 17
        for ( portIndex = 16; portIndex <= 17; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort 16, 17
            for ( outPortNum = 16; outPortNum <= 17; outPortNum++ )
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
                                             1 << (4*(outPortNum - 16)), // vcModeValid17_16
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

void basicE3600Config4::makeGangedLinkTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE3600Config4::makeAccessPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          16,       0,        0,        3,        DC_COUPLED);
        makeOneAccessPort(0,        0,          17,       0,        0,        2,        DC_COUPLED);
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config4::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        return;
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config4::makeOneWillow( int nodeIndex, int willowIndex )
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

void basicE3600Config4::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void basicE3600Config4::makeNodes()
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
basicE3600Config4::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config4::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config4::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config4::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
