#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "basicE3600Config3.h"
#include "fm_log.h"

basicE3600Config3::basicE3600Config3( fabricTopologyEnum topo ) : fabricConfig( topo )
{

};

basicE3600Config3::~basicE3600Config3()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void basicE3600Config3::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE3600Config3::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE3600Config3::makeGangedLinkTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE3600Config3::makeAccessPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          0,        0,        0,        5,         DC_COUPLED);
        makeOneAccessPort(0,        0,          1,        0,        0,        4,         DC_COUPLED);
        makeOneAccessPort(0,        0,          2,        0,        0,        2,         DC_COUPLED);
        makeOneAccessPort(0,        0,          3,        0,        0,        3,         DC_COUPLED);
        makeOneAccessPort(0,        0,          4,        0,        0,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          5,        0,        0,        0,         DC_COUPLED);
        makeOneAccessPort(0,        0,          8,        0,        0,        13,        DC_COUPLED);
        makeOneAccessPort(0,        0,          9,        0,        0,        12,        DC_COUPLED);
        makeOneAccessPort(0,        0,          10,       0,        0,        10,        DC_COUPLED);
        makeOneAccessPort(0,        0,          11,       0,        0,        11,        DC_COUPLED);
        makeOneAccessPort(0,        0,          12,       0,        0,        9,         DC_COUPLED);
        makeOneAccessPort(0,        0,          13,       0,        0,        8,         DC_COUPLED);
        makeOneAccessPort(0,        0,          16,       0,        0,        16,        DC_COUPLED);
        makeOneAccessPort(0,        0,          17,       0,        0,        17,        DC_COUPLED);
    }
    else
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config3::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //               nodeIndex willowIndex portIndex farNodeID farSwitchID farPortNum portMode
        makeOneTrunkPort(0,        0,          6,        0,        0,          15,        AC_COUPLED);
        makeOneTrunkPort(0,        0,          7,        0,        0,          14,        AC_COUPLED);
        makeOneTrunkPort(0,        0,          14,       0,        0,          7,         AC_COUPLED);
        makeOneTrunkPort(0,        0,          15,       0,        0,          6,         AC_COUPLED);
    }
    else
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config3::makeOneWillow( int nodeIndex, int willowIndex )
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

void basicE3600Config3::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void basicE3600Config3::makeNodes()
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
            // node 0 has 0 GPUs, 1 Willow
            makeOneNode( nodeIndex, 0, 1);
            break;

        default:
            FM_LOG_ERROR( "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}
