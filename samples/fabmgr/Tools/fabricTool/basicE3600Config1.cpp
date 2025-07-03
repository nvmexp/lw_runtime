#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "basicE3600Config1.h"
#include "fm_log.h"

basicE3600Config1::basicE3600Config1( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x6AC << 34;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 4;

    gpuFabricAddrBase[1]  = (uint64_t)0x0AB << 34;
    gpuFabricAddrRange[1] = FAB_ADDR_RANGE_16G * 4;
};

basicE3600Config1::~basicE3600Config1()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void basicE3600Config1::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            gpu   = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index & 0xFFFB ) << 34; // clear bit 36 for GPU
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        //mappedAddr,    // 64 bits fabric address
                                        0,
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            gpu   = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index & 0xFFFB ) << 34; // clear bit 36 for GPU
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        //mappedAddr,    // 64 bits fabric address
                                        0,
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    }
}

void basicE3600Config1::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, index, enpointID, outPortNum;
    accessPort *outPort;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            // gpuIndex 1 is connected to outPort 8 to 13
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
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
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                 nodeIndex, willowIndex, outPortNum);
                }
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort 0 to 5
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
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
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                 nodeIndex, willowIndex, outPortNum);
                }
            }
        }

    }
}

void basicE3600Config1::makeGangedLinkTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE3600Config1::makeAccessPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          0,        0,        0,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          1,        0,        0,        0,         DC_COUPLED);
        makeOneAccessPort(0,        0,          2,        0,        0,        5,         DC_COUPLED);
        makeOneAccessPort(0,        0,          3,        0,        0,        4,         DC_COUPLED);
        makeOneAccessPort(0,        0,          4,        0,        0,        2,         DC_COUPLED);
        makeOneAccessPort(0,        0,          5,        0,        0,        3,         DC_COUPLED);
        makeOneAccessPort(0,        0,          8,        0,        1,        2,         DC_COUPLED);
        makeOneAccessPort(0,        0,          9,        0,        1,        3,         DC_COUPLED);
        makeOneAccessPort(0,        0,          10,       0,        1,        4,         DC_COUPLED);
        makeOneAccessPort(0,        0,          11,       0,        1,        5,         DC_COUPLED);
        makeOneAccessPort(0,        0,          12,       0,        1,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          13,       0,        1,        0,         DC_COUPLED);
    }
    else
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config1::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //               nodeIndex willowIndex portIndex farNodeID farSwitchID farPortNum
        // No trunk ports
    }
    else
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config1::makeOneWillow( int nodeIndex, int willowIndex )
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
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config1::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void basicE3600Config1::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    for (nodeIndex = 0; nodeIndex < BASIC_E3600_NUM_NODES; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        // set up node IP address
        //nodeip << "192.168.0." << (nodeIndex + 1);
        //nodes[nodeIndex]->set_ipaddress( nodeip.str().c_str() );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 2 GPUs, 1 Willow
            makeOneNode( nodeIndex, 2, 1);
            break;

        default:
            FM_LOG_ERROR("Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

