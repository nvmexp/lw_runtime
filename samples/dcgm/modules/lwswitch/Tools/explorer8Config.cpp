#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <string.h>

#include "explorer8Config.h"

	static HWLink8 HWLinks[48] = {
                                        {GB1_SW1_G1,   4,   GB1__GPU4,     1},
                                        {GB1_SW1_G1,   5,   GB1__GPU1,     1},
                                        {GB1_SW1_G1,   12,  GB1__GPU8,     1},
                                        {GB1_SW1_G1,   13,  GB1__GPU5,     1},
                                        {GB1_SW1_G1,   14,  GB1__GPU6,     1},
                                        {GB1_SW1_G1,   15,  GB1__GPU7,     1},
                                        {GB1_SW1_G1,   16,  GB1__GPU2,     1},
                                        {GB1_SW1_G1,   17,  GB1__GPU3,     1},
                                        {GB1_SW2_G2,   4,   GB1__GPU3,     5},
                                        {GB1_SW2_G2,   5,   GB1__GPU8,     5},
                                        {GB1_SW2_G2,   6,   GB1__GPU1,     5},
                                        {GB1_SW2_G2,   7,   GB1__GPU4,     4},
                                        {GB1_SW2_G2,   12,  GB1__GPU6,     4},
                                        {GB1_SW2_G2,   13,  GB1__GPU7,     4},
                                        {GB1_SW2_G2,   16,  GB1__GPU5,     5},
                                        {GB1_SW2_G2,   17,  GB1__GPU2,     4},
                                        {GB1_SW3_G3,   4,   GB1__GPU6,     3},
                                        {GB1_SW3_G3,   5,   GB1__GPU7,     3},
                                        {GB1_SW3_G3,   6,   GB1__GPU2,     2},
                                        {GB1_SW3_G3,   7,   GB1__GPU4,     2},
                                        {GB1_SW3_G3,   13,  GB1__GPU5,     3},
                                        {GB1_SW3_G3,   14,  GB1__GPU8,     4},
                                        {GB1_SW3_G3,   15,  GB1__GPU1,     3},
                                        {GB1_SW3_G3,   17,  GB1__GPU3,     2},
                                        {GB1_SW4_G4,   4,   GB1__GPU8,     0},
                                        {GB1_SW4_G4,   5,   GB1__GPU6,     0},
                                        {GB1_SW4_G4,   12,  GB1__GPU7,     0},
                                        {GB1_SW4_G4,   13,  GB1__GPU5,     0},
                                        {GB1_SW4_G4,   14,  GB1__GPU3,     0},
                                        {GB1_SW4_G4,   15,  GB1__GPU2,     0},
                                        {GB1_SW4_G4,   16,  GB1__GPU1,     0},
                                        {GB1_SW4_G4,   17,  GB1__GPU4,     0},
                                        {GB1_SW5_G5,   5,   GB1__GPU7,     2},
                                        {GB1_SW5_G5,   6,   GB1__GPU3,     4},
                                        {GB1_SW5_G5,   7,   GB1__GPU2,     5},
                                        {GB1_SW5_G5,   12,  GB1__GPU8,     3},
                                        {GB1_SW5_G5,   13,  GB1__GPU5,     2},
                                        {GB1_SW5_G5,   14,  GB1__GPU6,     5},
                                        {GB1_SW5_G5,   15,  GB1__GPU1,     4},
                                        {GB1_SW5_G5,   17,  GB1__GPU4,     5},
                                        {GB1_SW6_G6,   5,   GB1__GPU6,     2},
                                        {GB1_SW6_G6,   6,   GB1__GPU1,     2},
                                        {GB1_SW6_G6,   7,   GB1__GPU4,     3},
                                        {GB1_SW6_G6,   12,  GB1__GPU2,     3},
                                        {GB1_SW6_G6,   13,  GB1__GPU3,     3},
                                        {GB1_SW6_G6,   14,  GB1__GPU5,     4},
                                        {GB1_SW6_G6,   15,  GB1__GPU8,     2},
                                        {GB1_SW6_G6,   17,  GB1__GPU7,     5},
		                             };
                            



explorer8Config::explorer8Config( fabricTopologyEnum topo ) : fabricConfig( topo )
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

explorer8Config::~explorer8Config() 
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void explorer8Config::makeIngressReqTable( int nodeIndex, int willowIndex ) 
{
    int32_t gpuEndpointId, portIndex, portGPU, peerGPU, peerGPULink, gpuIndex, linkBase, linkIndex, index, routeIndex;
    int64_t mappedAddr, range, i;
    GPU *gpu;
    uint32_t valid7_0[8]   = {0};  // vc_valid bits for each GPU 
    uint32_t valid15_8[8]  = {0};
    uint32_t valid17_16[8] = {0};
    linkBase = willowIndex * 8;

    // first poplulate the vcValid array with correct egress for each ingress

    for ( gpuIndex = 0; gpuIndex < 8; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + 8; portIndex++ )
        { 
            //find the first port for the current GPU
            if ( HWLinks[portIndex].GPUIndex == gpuIndex )
            {
                if (HWLinks[portIndex].willowPort < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * HWLinks[portIndex].willowPort);
                } 
                else if (HWLinks[portIndex].willowPort < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (HWLinks[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[gpuIndex] = 1 << (4 * (HWLinks[portIndex].willowPort - 16) );
                }
                break;
            } else {
                continue;
            }
        }
    }
             
    // now build the request table for each port       
    for ( portIndex = 0; portIndex <= 7; portIndex++ )
    {
        for ( gpuIndex = 0; gpuIndex <= 7; gpuIndex++ )
        {
            if ( HWLinks[linkBase + portIndex].GPUIndex == gpuIndex )
            {
                // skip over the loopback case
                continue;
            }
            gpu = gpus[nodeIndex][gpuIndex];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                //Everything in Explorer 8 is last hop and mapped back to 0
                mappedAddr = i << 34;

                makeOneIngressReqEntry( nodeIndex,                          // nodeIndex
                                        willowIndex,                        // willowIndex
                                        HWLinks[linkBase + portIndex].willowPort,                          
                                                                            // portIndex
                                        index,                              // ingress req table index
                                        mappedAddr,                         // 64 bits fabric address
                                        0,                                  // routePolicy
                                        valid7_0[gpuIndex],                 // vcModeValid7_0
                                        valid15_8[gpuIndex],                // vcModeValid15_8, port 8 to 13
                                        valid17_16[gpuIndex],               // vcModeValid17_16
                                        1);                                 // entryValid
            }
        }
    }
};

void explorer8Config::makeIngressRespTable( int nodeIndex, int willowIndex ) 
{
    int gpuIndex, portIndex, i, index, enpointID, outPortNum, linkBase, subIndex, routeIndex;
    accessPort *outPort;
    unsigned int rlid[8];
    uint32_t valid7_0[8]   = {0};  // vc_valid bits for each GPU 
    uint32_t valid15_8[8]  = {0};
    uint32_t valid17_16[8] = {0};
    linkBase = willowIndex * 8;

    // first poplulate the vcValid array with correct egress for each ingress

    for ( gpuIndex = 0; gpuIndex < 8; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + 8; portIndex++ )
        {
            if ( HWLinks[portIndex].GPUIndex == gpuIndex )
            { 
                if (HWLinks[portIndex].willowPort < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * HWLinks[portIndex].willowPort);
                } 
                else if (HWLinks[portIndex].willowPort < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (HWLinks[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[gpuIndex] = 1 << (4 * (HWLinks[portIndex].willowPort - 16) );
                }

                // also record the egress, and requester link ID of the connecting port
                rlid[gpuIndex] = 6 * gpuIndex + HWLinks[portIndex].GPUPort;
                
                break;
            } else {
                continue;
            }
        }
    }
                
    for ( portIndex = 0; portIndex < 8; portIndex++ )
    { 
        for ( gpuIndex = 0; gpuIndex < 8; gpuIndex++ )
        {
        // each egress serves 1 ingress from each of the other GPUs
            if ( HWLinks[linkBase + portIndex].GPUIndex == gpuIndex )
            { 
                continue;
            }
            else
            {
                index = rlid[gpuIndex];
 
                makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                         willowIndex,                               // willowIndex
                                         HWLinks[linkBase + portIndex].willowPort,                          
                                                                                    // portIndex
                                         index,                                     // Ingress resq table index
                                         0,                                         // routePolicy
                                         valid7_0[gpuIndex],                        // vcModeValid7_0
                                         valid15_8[gpuIndex],                       // vcModeValid15_8, port 8 to 13
                                         valid17_16[gpuIndex],                      // vcModeValid17_16
                                         1);                                        // entryValid

            }
        }
    }

}

void explorer8Config::makeGangedLinkTable( int nodeIndex, int willowIndex ) 
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

void explorer8Config::makeAccessPorts( int nodeIndex, int willowIndex ) 
{
    int i;
    int baseIndex = willowIndex * 8;

    for ( i = 0; i < 8; i++ )   
    {
        makeOneAccessPort(0,                                    // nodeIndex 
                          willowIndex,                          // willowIndex
                          HWLinks[baseIndex + i].willowPort,    // portIndex
                          0,                                    // farNodeID        
                          HWLinks[baseIndex + i].GPUIndex,      // farPeerID         
                          HWLinks[baseIndex + i].GPUPort,       // farPortNum        
                          DC_COUPLED);                          // portMode
    }
}

void explorer8Config::makeTrunkPorts( int nodeIndex, int willowIndex ) 
{
    return;
}

void explorer8Config::makeOneWillow( int nodeIndex, int willowIndex ) 
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( ( (nodeIndex == 0) && (willowIndex >= 0) ) &&
         ( (nodeIndex == 0) && (willowIndex <= 5) ) )
    {
        switches[nodeIndex][willowIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][willowIndex]->set_ecid( ecid.str().c_str() );

	//For 8-GPU Explorer, physical ID equals index plus 0x08.
        switches[nodeIndex][willowIndex]->set_physicalid( willowIndex + 0x08 );

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

void explorer8Config::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void explorer8Config::makeNodes() 
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
            // node 0 has 8 GPUs, 6 Willows
            makeOneNode( nodeIndex, 8, 6);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
explorer8Config::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
explorer8Config::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorer8Config::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorer8Config::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
