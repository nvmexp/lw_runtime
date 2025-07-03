#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <string.h>

#include "explorerLoop.h"

	static HWLink HWLinks[48] = {
		                    {U1LWP1,0,LWL_GPU1,1},
		                    {U1LWP1,1,LWL_GPU1,0},
		                    {U1LWP1,2,LWL_GPU8,3},
		                    {U1LWP1,3,LWL_GPU8,2},
		                    {U1LWP1,4,LWL_GPU4,0},
		                    {U1LWP1,5,LWL_GPU4,1},
		                    {U1LWP1,6,LWL_GPU5,2},
		                    {U1LWP1,7,LWL_GPU5,3},
		                    {U1LWP1,8,LWL_GPU3,0},
		                    {U1LWP1,9,LWL_GPU3,1},
		                    {U1LWP1,10,LWL_GPU6,2},
		                    {U1LWP1,11,LWL_GPU6,3},
		                    {U1LWP1,12,LWL_GPU2,1},
		                    {U1LWP1,13,LWL_GPU2,0},
		                    {U1LWP1,14,LWL_GPU7,3},
		                    {U1LWP1,15,LWL_GPU7,2},
		                    {U1LWP2,0,LWL_GPU5,5},
		                    {U1LWP2,1,LWL_GPU5,4},
		                    {U1LWP2,2,LWL_GPU7,5},
		                    {U1LWP2,3,LWL_GPU7,4},
		                    {U1LWP2,4,LWL_GPU3,4},
		                    {U1LWP2,5,LWL_GPU3,5},
		                    {U1LWP2,6,LWL_GPU1,4},
		                    {U1LWP2,7,LWL_GPU1,5},
		                    {U1LWP2,8,LWL_GPU6,4},
		                    {U1LWP2,9,LWL_GPU6,5},
		                    {U1LWP2,10,LWL_GPU8,4},
		                    {U1LWP2,11,LWL_GPU8,5},
		                    {U1LWP2,12,LWL_GPU4,5},
		                    {U1LWP2,13,LWL_GPU4,4},
		                    {U1LWP2,14,LWL_GPU2,5},
		                    {U1LWP2,15,LWL_GPU2,4},
		                    {U1LWP3,0,LWL_GPU7,1},
		                    {U1LWP3,1,LWL_GPU7,0},
		                    {U1LWP3,2,LWL_GPU2,3},
		                    {U1LWP3,3,LWL_GPU2,2},
		                    {U1LWP3,4,LWL_GPU6,0},
		                    {U1LWP3,5,LWL_GPU6,1},
		                    {U1LWP3,6,LWL_GPU3,2},
		                    {U1LWP3,7,LWL_GPU3,3},
		                    {U1LWP3,8,LWL_GPU5,0},
		                    {U1LWP3,9,LWL_GPU5,1},
		                    {U1LWP3,10,LWL_GPU4,2},
		                    {U1LWP3,11,LWL_GPU4,3},
		                    {U1LWP3,12,LWL_GPU8,1},
		                    {U1LWP3,13,LWL_GPU8,0},
		                    {U1LWP3,14,LWL_GPU1,3},
		                    {U1LWP3,15,LWL_GPU1,2}
		                    };
                            



explorerLoop::explorerLoop( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x000 << 36;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[1]  = (uint64_t)0x001 << 36;
    gpuFabricAddrRange[1] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[2]  = (uint64_t)0x002 << 36;
    gpuFabricAddrRange[2] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[3]  = (uint64_t)0x003 << 36;
    gpuFabricAddrRange[3] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[4]  = (uint64_t)0x004 << 36;
    gpuFabricAddrRange[4] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[5]  = (uint64_t)0x005 << 36;
    gpuFabricAddrRange[5] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[6]  = (uint64_t)0x006 << 36;
    gpuFabricAddrRange[6] = FAB_ADDR_RANGE_16G * 2;

    gpuFabricAddrBase[7]  = (uint64_t)0x007 << 36;
    gpuFabricAddrRange[7] = FAB_ADDR_RANGE_16G * 2;
};

explorerLoop::~explorerLoop()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void explorerLoop::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, portGPU, peerGPU, peerGPULink, gpuIndex, linkBase, linkIndex, index, routeIndex;
    int64_t mappedAddr, range, i;
    GPU *gpu;
    vcValid valid[16];  // vc_valid bits for each GPU times two ports
    linkBase = willowIndex * 16;

    // first poplulate the vcValid array with correct egress for each ingress
    // this is loop back, so it's a trivial case.

    for ( portIndex = 0; portIndex < 16; portIndex++ )
    { 
        //note we are using the fact that they are always paired
        valid[portIndex].port0_15 = 1L << (4 * portIndex);
    }
             
    // now build the request table for each port       

    for ( portIndex = 0; portIndex <= 15; portIndex++ )
    {
   
        gpuIndex = HWLinks[linkBase + portIndex].GPUIndex;
        gpu = gpus[nodeIndex][gpuIndex];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            //Everything in Explorer is last hop and mapped back to 0
            mappedAddr = i << 34;

            makeOneIngressReqEntry( nodeIndex,                          // nodeIndex
                                    willowIndex,                        // willowIndex
                                    portIndex,                          // portIndex
                                    index,                              // ingress req table index
                                    mappedAddr,                         // 64 bits fabric address
                                    0,                                  // routePolicy
                                    valid[portIndex].hwRegs.port0_7,    // vcModeValid7_0
                                    valid[portIndex].hwRegs.port8_15,  // vcModeValid15_8, port 8 to 13
                                    0x00000000,                         // vcModeValid17_16
                                    1);                                 // entryValid
        }
    }
};

void explorerLoop::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, i, index, enpointID, outPortNum, linkBase, subIndex, routeIndex;
    accessPort *outPort;
    vcValid valid[16];  // vc_valid bits for each GPU times two ports
    unsigned int rlid[16];

    linkBase = willowIndex * 16;

    // first poplulate the vcValid array with correct egress for each ingress
    // this is loop back, so it's a trivial case.

    for ( portIndex = 0; portIndex < 16; portIndex++ )
    { 
        //note we are using the fact that they are always paired
        valid[portIndex].port0_15 = 1L << (4 * portIndex);
    }
             
                
    for ( portIndex = 0; portIndex < 16; portIndex++ )
    { 
        gpuIndex = HWLinks[linkBase + portIndex].GPUIndex;

        // index is the requesterLinkId
        index = gpuIndex * 6 + HWLinks[linkBase + portIndex].GPUPort;

        makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                 willowIndex,                               // willowIndex
                                 portIndex,                                 // portIndex
                                 index,                                     // Ingress resq table index
                                 0,                                         // routePolicy
                                 valid[portIndex].hwRegs.port0_7,           // vcModeValid7_0
                                 valid[portIndex].hwRegs.port8_15,          // vcModeValid15_8, port 8 to 13
                                 0x00000000,                                // vcModeValid17_16
                                 1);                                        // entryValid

    }

}

void explorerLoop::makeGangedLinkTable( int nodeIndex, int willowIndex )
{
    int32_t portIndex, i;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            for ( i = 0; i < GANGED_LINK_TABLE_SIZE; i++ )
            {
                makeOneGangedLinkEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        i,             // ganged link table index
                                        i) ;           // ganged link table data
            }
        }
    }
}

void explorerLoop::makeAccessPorts( int nodeIndex, int willowIndex )
{
    int i;
    int baseIndex = willowIndex * 16;

    for ( i = 0; i < 16; i++ )   
    {
            
        makeOneAccessPort(0,                                    // nodeIndex 
                          willowIndex,                          // willowIndex
                          i,                                    // portIndex
                          0,                                    // farNodeID        
                          HWLinks[baseIndex + i].GPUIndex,      // farPeerID         
                          HWLinks[baseIndex + i].GPUPort,       // farPortNum        
                          DC_COUPLED);                          // portMode
    }
}

void explorerLoop::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    return;
}

void explorerLoop::makeOneWillow( int nodeIndex, int willowIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( ( (nodeIndex == 0) && (willowIndex == 0) ) ||
         ( (nodeIndex == 0) && (willowIndex == 1) ) || 
         ( (nodeIndex == 0) && (willowIndex == 2) ) )
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

void explorerLoop::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void explorerLoop::makeNodes()
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
            // node 0 has 8 GPUs, 3 Willow
            makeOneNode( nodeIndex, 8, 3);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
explorerLoop::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
explorerLoop::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorerLoop::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorerLoop::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
