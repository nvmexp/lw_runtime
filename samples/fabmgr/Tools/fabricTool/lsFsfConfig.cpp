#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "fabricConfig.h"
#include "lsFsfConfig.h"
#include "FMDeviceProperty.h"

lsFsfConfig::lsFsfConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    mNumGpus = 2;
    mNumSwitches = 1;

    for (uint32_t targetId = 0; targetId < mNumGpus; targetId++) {
        gpuFabricAddrBase[targetId]  = FMDeviceProperty::getGpaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFabricAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuGpaEgmAddrBase[targetId]  = FMDeviceProperty::getGpaEgmFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuGpaEgmAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuFlaAddrBase[targetId]  = FMDeviceProperty::getFlaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFlaAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);
    }
};

lsFsfConfig::~lsFsfConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void lsFsfConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;
    uint32_t targetId;

    // Access port
    for ( portIndex = 0; portIndex <= 11; portIndex++ )
    {
        for (uint32_t targetId = 0; targetId < mNumGpus; targetId++ )
        {
            gpu   = gpus[nodeIndex][targetId];

            // GPA map slot
            index = FMDeviceProperty::getGpaRemapIndexFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
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
                               targetId,            // target ID
                               1,                   // remapFlags = remap GPA address, AddrType[1:0] 2’b10 Map Slot
                               0,                   // irl select
                               0,                   // FAM swizzle
                               0,                   // FAM mult2
                               0,                   // FAM plane select
                               EXTENDED_RANGE_B);   // remap table select

            // FLA map slot
            index = FMDeviceProperty::getFlaRemapIndexFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
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
                               targetId,            // target ID
                               1,                   // remapFlags = remap FLA address, AddrType[1:0] 2’b10 Map Slot
                               0,                   // irl select
                               0,                   // FAM swizzle
                               0,                   // FAM mult2
                               0,                   // FAM plane select
                               NORMAL_RANGE);       // remap table select
        }
    }
}

void lsFsfConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS], portCount;
    uint32_t targetId;

    // Access port 0 to 5
    for ( portIndex = 0; portIndex <= 5; portIndex++ )
    {
        // connected to GPU 0, going to GPU 1 targetId 1
        targetId  = 1;
        portCount = 1;
        vcMap[0]  = 0;

        // GPU 1 are connected to port 6 to 11
        egressPort[0] = portIndex + 6;

        makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                              swIndex,          // swIndex
                              portIndex,        // portIndex
                              targetId,         // RID Route table index, target ID
                              1,                // entry valid
                              0,                // rmod (no special routing)
                              portCount,        // number of ports
                              vcMap,            // pointer to array of VC controls
                              egressPort);      // pointer to array of ports
    }

    // Access port 6 to 11
    for ( portIndex = 6; portIndex <= 11; portIndex++ )
    {
        // connected to GPU 1, going to GPU 0 targetId 0
        targetId  = 0;
        portCount = 1;
        vcMap[0]  = 0;

        // GPU 0 are connected to port 0 to 5
        egressPort[0] = portIndex - 6;

        makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                              swIndex,          // swIndex
                              portIndex,        // portIndex
                              targetId,         // RID Route table index, target ID
                              1,                // entry valid
                              0,                // rmod (no special routing)
                              portCount,        // number of ports
                              vcMap,            // pointer to array of VC controls
                              egressPort);      // pointer to array of ports
    }
}

void lsFsfConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    // Access port 0 to 11
    for ( uint32_t portIndex = 0; portIndex <= 11; portIndex++ )
    {
        for (uint32_t targetId = 0; targetId < mNumGpus; targetId++ )
        {
            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   targetId,         // RLAN Route table index
                                   1,                // entry valid
                                   0,                // group count
                                   NULL,             // group select array
                                   NULL );           // group size array
        }
    }
}

void lsFsfConfig::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void lsFsfConfig::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    // access port 0 - 5 connected to GPU 0
    farPeerID = 0;
    farPeerTargetID  = 0;

    //                nodeIndex swIndex   portIndex  farNodeID farPeerID  farPortNum portMode    farPeerTargetID  rlanID
    makeOneAccessPort(0,        0,        0,         0,        farPeerID, 0,         DC_COUPLED, farPeerTargetID, 0);
    makeOneAccessPort(0,        0,        1,         0,        farPeerID, 1,         DC_COUPLED, farPeerTargetID, 1);
    makeOneAccessPort(0,        0,        2,         0,        farPeerID, 2,         DC_COUPLED, farPeerTargetID, 2);
    makeOneAccessPort(0,        0,        3,         0,        farPeerID, 3,         DC_COUPLED, farPeerTargetID, 3);
    makeOneAccessPort(0,        0,        4,         0,        farPeerID, 4,         DC_COUPLED, farPeerTargetID, 4);
    makeOneAccessPort(0,        0,        5,         0,        farPeerID, 5,         DC_COUPLED, farPeerTargetID, 5);

    // access port 6 - 11 connected to GPU 1
    farPeerID = 1;
    farPeerTargetID = 1;
    //                nodeIndex swIndex   portIndex  farNodeID farPeerID  farPortNum portMode    farPeerTargetID  rlanID
    makeOneAccessPort(0,        0,        6,         0,        farPeerID, 0,         DC_COUPLED, farPeerTargetID, 0);
    makeOneAccessPort(0,        0,        7,         0,        farPeerID, 1,         DC_COUPLED, farPeerTargetID, 1);
    makeOneAccessPort(0,        0,        8,         0,        farPeerID, 2,         DC_COUPLED, farPeerTargetID, 2);
    makeOneAccessPort(0,        0,        9,         0,        farPeerID, 3,         DC_COUPLED, farPeerTargetID, 3);
    makeOneAccessPort(0,        0,        10,        0,        farPeerID, 4,         DC_COUPLED, farPeerTargetID, 4);
    makeOneAccessPort(0,        0,        11,        0,        farPeerID, 5,         DC_COUPLED, farPeerTargetID, 5);
}

void lsFsfConfig::makeTrunkPorts( int nodeIndex, int swIndex )
{
    // no trunk port in FSF
    return;
}

void lsFsfConfig::makeOneLwswitch( int nodeIndex, int swIndex )
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
        printf("Invalid LS10 nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void lsFsfConfig::makeOneNode( int nodeIndex, int gpuNum, int lsNum )
{
    int i;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, i);
    }

    // Add Switches
    for ( i = 0; i < lsNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }
}

void lsFsfConfig::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 2 GPUs, 1 Laguna
            mNumGpus = 2;
            mNumSwitches = 1;
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void lsFsfConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void lsFsfConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void lsFsfConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
