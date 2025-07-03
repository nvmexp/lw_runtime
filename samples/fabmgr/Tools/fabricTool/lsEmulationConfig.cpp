#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "fabricConfig.h"
#include "lsEmulationConfig.h"
#include "FMDeviceProperty.h"

lsEmulationConfig::lsEmulationConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    mNumGpus = 1;
    mNumSwitches = 1;
    mUseTrunkLoopback = false;

    for (uint32_t targetId = 0; targetId < mNumGpus; targetId++) {
        gpuFabricAddrBase[targetId]  = FMDeviceProperty::getGpaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFabricAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuGpaEgmAddrBase[targetId]  = FMDeviceProperty::getGpaEgmFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuGpaEgmAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuFlaAddrBase[targetId]  = FMDeviceProperty::getFlaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFlaAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);
    }
};

lsEmulationConfig::~lsEmulationConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void lsEmulationConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;
    uint32_t targetId;

    // Access port
    for ( portIndex = 0; portIndex <= 0; portIndex++ )
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

void lsEmulationConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    int gpuIndex, portIndex, egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS], portCount;
    uint32_t targetId;

    // Access port 0 connect to GPU 0 port 0

    // connected to GPU 0, going to GPU 1 targetId 0
    portIndex = 0;
    targetId  = 0;
    portCount = 1;
    vcMap[0]  = 0;

    // Access port 0
    if (mUseTrunkLoopback) {
        egressPort[0] = 1; // egress Trunk port 1
    } else {
        egressPort[0] = 0; // egress Access port 0
    }

    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                          swIndex,          // swIndex
                          portIndex,        // portIndex
                          targetId,         // RID Route table index, target ID
                          1,                // entry valid
                          0,                // rmod (no special routing)
                          portCount,        // number of ports
                          vcMap,            // pointer to array of VC controls
                          egressPort);      // pointer to array of ports

    // Trunk port 1
    portIndex = 1;
    egressPort[0] = 2; // egress Trunk port 2
    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                          swIndex,          // swIndex
                          portIndex,        // portIndex
                          targetId,         // RID Route table index, target ID
                          1,                // entry valid
                          0,                // rmod (no special routing)
                          portCount,        // number of ports
                          vcMap,            // pointer to array of VC controls
                          egressPort);      // pointer to array of ports

    // Trunk port 2
    portIndex = 2;
    egressPort[0] = 3; // egress Trunk port 3
    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                          swIndex,          // swIndex
                          portIndex,        // portIndex
                          targetId,         // RID Route table index, target ID
                          1,                // entry valid
                          0,                // rmod (no special routing)
                          portCount,        // number of ports
                          vcMap,            // pointer to array of VC controls
                          egressPort);      // pointer to array of ports

    // Trunk port 3
    portIndex = 3;
    egressPort[0] = 0; // egress Access port 0 to go back to the GPU
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

void lsEmulationConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        // Access port 0, Trunk port 1, 2, 3
        for ( uint32_t portIndex = 0; portIndex <= 3; portIndex++ )
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
}

void lsEmulationConfig::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void lsEmulationConfig::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    farPeerID = 0;
    farPeerTargetID = 0;

    //                nodeIndex swIndex     portIndex farNodeID farPeerID farPortNum portMode    farPeerTargetID  rlanID
    makeOneAccessPort(0,        0,          0,        0,        farPeerID,  0,       DC_COUPLED, farPeerTargetID, 0);
}

void lsEmulationConfig::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        //               nodeIndex swIndex portIndex farNodeID farSwitchID farPortNum portMode
        makeOneTrunkPort(0,        0,      1,        0,        0,          0,         DC_COUPLED);
        makeOneTrunkPort(0,        0,      2,        0,        0,          3,         DC_COUPLED);
        makeOneTrunkPort(0,        0,      3,        0,        0,          2,         DC_COUPLED);
    }
    else
    {
        printf("Invalid LimeRock nodeIndex %d swIndex %d.\n", nodeIndex, swIndex);
    }
}

void lsEmulationConfig::makeOneLwswitch( int nodeIndex, int swIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << swIndex;

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

void lsEmulationConfig::makeOneNode( int nodeIndex, int gpuNum, int lsNum )
{
    int i;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, i);
    }

    // Add LS10s
    for ( i = 0; i < lsNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }
}

void lsEmulationConfig::makeNodes()
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
            // node 0 has 1 GPUs, 1 LS10
            makeOneNode( nodeIndex, 1, 1);
            break;

        default:
            printf("Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void lsEmulationConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void lsEmulationConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void lsEmulationConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
