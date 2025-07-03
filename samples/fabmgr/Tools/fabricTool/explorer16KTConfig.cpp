#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <unistd.h>
#include <string.h>
#include <strings.h>

#include "explorer16common.h"
#include "explorer16KTConfig.h"
#include "fm_log.h"


#define EXPLORER16_KT_NUM_NODES     2
#define EXPLORER16_KT_NUM_SWITCH_PER_NODE       6
#define EXPLORER16_KT_NUM_LINKS_PER_GPU         6
#define EXPLORER16_KT_NUM_GPU_PER_NODE          8
explorer16KTConfig::explorer16KTConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t i;
    for (i = 0; i < 16; i++) 
    {
        gpuFabricAddrBase[i]  = i << 36;
        gpuFabricAddrRange[i] = FAB_ADDR_RANGE_16G * 2;
    }
};

explorer16KTConfig::~explorer16KTConfig() 
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void explorer16KTConfig::makeIngressReqTable( int nodeIndex, int willowIndex ) 
{
    int32_t gpuEndpointId, portIndex, portGPU, peerGPU, peerGPULink;
    int32_t gpuIndex, linkBase, linkIndex, index, routeIndex;
    int64_t mappedAddr, range, i;
    GPU *gpu;
    uint32_t valid7_0[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];  // vc_valid bits for each GPU 
    uint32_t valid15_8[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];
    uint32_t valid17_16[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];
    int nodeIndexFar = (nodeIndex + 1 ) % 2;

    bzero(valid7_0, sizeof(valid7_0));
    bzero(valid15_8, sizeof(valid15_8));
    bzero(valid17_16, sizeof(valid17_16));

    linkBase = (nodeIndex * EXPLORER16_KT_NUM_SWITCH_PER_NODE * EXPLORER16_KT_NUM_GPU_PER_NODE) 
                + (willowIndex * EXPLORER16_KT_NUM_GPU_PER_NODE);
    // begin access ingress routing
    // first poplulate the vcValid array with correct egress for each access ingress on the local node

    for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
        { 
            //find the first port for the current GPU
            if ( HWLinks16[portIndex].GPUIndex == (gpuIndex + (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE)))
            {
                if (HWLinks16[portIndex].willowPort < EXPLORER16_KT_NUM_GPU_PER_NODE)
                {
                    valid7_0[nodeIndex][gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }
                break;
            } else {
                continue;
            }
        }
    }
    uint32_t *pTrunkPorts;
    if(nodeIndex == 0)
        pTrunkPorts = trunkPortsFar;
    else
        pTrunkPorts = trunkPortsNear;

    // for the far GPUs we can select the egress in order of appearance
    for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE ; gpuIndex++ )
    {
        if (pTrunkPorts[gpuIndex] < 8)
        {
            valid7_0[nodeIndexFar][gpuIndex] = 1 << (4 * pTrunkPorts[gpuIndex]);
        } 
        else if (pTrunkPorts[gpuIndex] < 16)
        {
            valid15_8[nodeIndexFar][gpuIndex] = 1 << (4 * (pTrunkPorts[gpuIndex] - 8) );
        }
        else //these are not used for trunk links but we may want this code elsewhere so keeping it generic.
        {
            valid17_16[nodeIndexFar][gpuIndex] = 1 << (4 * (pTrunkPorts[gpuIndex] - 16) );
        }
    }
             
    // now build the request table for each port for GPUs on this node      
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    {
        for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
        {
            if ( HWLinks16[linkBase + portIndex].GPUIndex == (gpuIndex + (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE)) )
            {
                // skip over the loopback case
                continue;
            }
            gpu = gpus[nodeIndex][gpuIndex];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                //Everything in the local board is last hop and mapped back to 0
                mappedAddr = i << 34;

                makeOneIngressReqEntry( nodeIndex,                                  // nodeIndex
                                        willowIndex,                                // willowIndex
                                        HWLinks16[linkBase + portIndex].willowPort, // portIndex
                                        index,                                      // ingress req table index
                                        mappedAddr,                                 // 64 bits fabric address
                                        0,                                          // routePolicy
                                        valid7_0[nodeIndex][gpuIndex],              // vcModeValid7_0
                                        valid15_8[nodeIndex][gpuIndex],             // vcModeValid15_8, port 8 to 13
                                        valid17_16[nodeIndex][gpuIndex],            // vcModeValid17_16
                                        1);                                         // entryValid
            }
        }
    }
    
    // now build the request table for each port for GPUs on the far node
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    {
        for ( gpuIndex = 0; gpuIndex <  EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
        {
            gpu = gpus[nodeIndexFar][gpuIndex];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();
            
            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                //Trunks have to preserve target address
                mappedAddr = gpu->fabricaddrbase() + (i << 34);

                makeOneIngressReqEntry( nodeIndex,                                  // nodeIndex
                                        willowIndex,                                // willowIndex
                                        HWLinks16[linkBase + portIndex].willowPort, // portIndex
                                        index,                                      // ingress req table index
                                        mappedAddr,                                 // 64 bits fabric address
                                        0,                                          // routePolicy
                                        valid7_0[nodeIndexFar][gpuIndex],           // vcModeValid7_0
                                        valid15_8[nodeIndexFar][gpuIndex],          // vcModeValid15_8, port 8 to 13
                                        valid17_16[nodeIndexFar][gpuIndex],         // vcModeValid17_16
                                        1);                                         // entryValid
            }
        
        }    
    }
    // end access ingress request routing
    
    // begin trunk ingress request routing
    
    // this could have been done as part of the first access port loop.
    // doing it separately to keep it simple.
    
    // each trunk port needs entries for GPUs on the near node only. 
    // build a list of which port goes to which GPU.
    for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
        { 
            //find the first port for the current GPU
            if ( HWLinks16[portIndex].GPUIndex == (gpuIndex + nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE))
            {
                if (HWLinks16[portIndex].willowPort < 8)
                {
                    valid7_0[nodeIndex][gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }
                break;
            } else {
                continue;
            }
        }
    }
    
    // note that any trunk port on the board could get data for any GPU, so all
    // trunk port ingress tables are identical. 
    
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    {
        for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
        {
            gpu = gpus[nodeIndex][gpuIndex];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                //Everything on the local board is last hop and mapped back to 0
                mappedAddr = i << 34;

                makeOneIngressReqEntry( nodeIndex,                          // nodeIndex
                                        willowIndex,                        // willowIndex
                                        trunkPortsNear[portIndex],          // portIndex
                                        index,                              // ingress req table index
                                        mappedAddr,                         // 64 bits fabric address
                                        0,                                  // routePolicy
                                        valid7_0[nodeIndex][gpuIndex],      // vcModeValid7_0
                                        valid15_8[nodeIndex][gpuIndex],     // vcModeValid15_8, port 8 to 13
                                        valid17_16[nodeIndex][gpuIndex],    // vcModeValid17_16
                                        1);                                 // entryValid
            }
        }
    }

};

void explorer16KTConfig::makeIngressRespTable( int nodeIndex, int willowIndex ) 
{
    int gpuIndex, portIndex, i, index, enpointID, outPortNum, linkBase, linkBaseFar, routeIndex;
    accessPort *outPort;
    unsigned int rlid[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];
    uint32_t valid7_0[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];  // vc_valid bits for each GPU 
    uint32_t valid15_8[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];
    uint32_t valid17_16[EXPLORER16_KT_NUM_NODES][EXPLORER16_KT_NUM_GPU_PER_NODE];
    int nodeIndexFar = (nodeIndex + 1 ) % 2;

    bzero(valid7_0, sizeof(valid7_0));
    bzero(valid15_8, sizeof(valid15_8));
    bzero(valid17_16, sizeof(valid17_16));

    linkBase = (nodeIndex * EXPLORER16_KT_NUM_SWITCH_PER_NODE * EXPLORER16_KT_NUM_GPU_PER_NODE)
                + (willowIndex * EXPLORER16_KT_NUM_GPU_PER_NODE);
    
    linkBaseFar = (nodeIndexFar * EXPLORER16_KT_NUM_SWITCH_PER_NODE * EXPLORER16_KT_NUM_GPU_PER_NODE)
                + (willowIndex * EXPLORER16_KT_NUM_GPU_PER_NODE);
    
    // begin access ingress response routing
    // first poplulate the vcValid array with correct egress for each access ingress

    for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
        {
            if ( HWLinks16[portIndex].GPUIndex == (gpuIndex + (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE)))
            { 
                if (HWLinks16[portIndex].willowPort < 8)
                {
                    valid7_0[nodeIndex][gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[nodeIndex][gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }

                // also record the egress, and requester link ID of the connecting port
                rlid[nodeIndex][gpuIndex] = (EXPLORER16_KT_NUM_LINKS_PER_GPU * EXPLORER16_KT_NUM_GPU_PER_NODE * nodeIndex ) 
                                            + (EXPLORER16_KT_NUM_LINKS_PER_GPU * gpuIndex) + HWLinks16[portIndex].GPUPort;
             
                break;
            } else {
                continue;
            }
        }
    }
    uint32_t *pTrunkPorts;
    if(nodeIndex == 0)
        pTrunkPorts = trunkPortsFar;
    else
        pTrunkPorts = trunkPortsNear;
    // for the far GPUs we can select the egress in order of appearance
    for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
    {
        for ( portIndex = linkBaseFar; portIndex < linkBaseFar + EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
        {
            if ( HWLinks16[portIndex].GPUIndex == (gpuIndex + (nodeIndexFar * EXPLORER16_KT_NUM_GPU_PER_NODE) )) 
            {
                if (pTrunkPorts[gpuIndex] < 8)
                {
                    valid7_0[nodeIndexFar][gpuIndex] = 1 << (4 * pTrunkPorts[gpuIndex]);
                } 
                else if (pTrunkPorts[gpuIndex] < 16)
                {
                    valid15_8[nodeIndexFar][gpuIndex] = 1 << (4 * (pTrunkPorts[gpuIndex] - 8) );
                }
                else //these are not used for trunk links but we may want this code elsewhere so keeping it generic.
                {
                    valid17_16[nodeIndexFar][gpuIndex] = 1 << (4 * (pTrunkPorts[gpuIndex] - 16) );
                }
                rlid[nodeIndexFar][gpuIndex] = (EXPLORER16_KT_NUM_LINKS_PER_GPU * EXPLORER16_KT_NUM_GPU_PER_NODE * nodeIndexFar )
                                                + (EXPLORER16_KT_NUM_LINKS_PER_GPU * gpuIndex) + HWLinks16[portIndex].GPUPort;
            }
        }
    }
             
                
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    { 
        for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
        {
        // each egress serves 1 ingress from each of the other GPUs
            if ( HWLinks16[linkBase + portIndex].GPUIndex == (gpuIndex + (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE)) )
            { 
                continue;
            }
            else
            {
                index = rlid[nodeIndex][gpuIndex];
 
                makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                         willowIndex,                               // willowIndex
                                         HWLinks16[linkBase + portIndex].willowPort,// portIndex
                                         index,                                     // Ingress req table index
                                         0,                                         // routePolicy
                                         valid7_0[nodeIndex][gpuIndex],             // vcModeValid7_0
                                         valid15_8[nodeIndex][gpuIndex],            // vcModeValid15_8, port 8 to 13
                                         valid17_16[nodeIndex][gpuIndex],           // vcModeValid17_16
                                         1);                                        // entryValid

            }
        }
    }
    // now build the response table for each port for GPUs on the far node
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    {
        for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; gpuIndex++ )
        {
            index = rlid[nodeIndexFar][gpuIndex];
            makeOneIngressRespEntry( nodeIndex,                                     // nodeIndex
                                     willowIndex,                                   // willowIndex
                                     HWLinks16[linkBase + portIndex].willowPort,    // portIndex
                                     index,                                         // Ingress req table index
                                     0,                                             // routePolicy
                                     valid7_0[nodeIndexFar][gpuIndex],              // vcModeValid7_0
                                     valid15_8[nodeIndexFar][gpuIndex],             // vcModeValid15_8, port 8 to 13
                                     valid17_16[nodeIndexFar][gpuIndex],            // vcModeValid17_16
                                     1);                                            // entryValid
        
        }
    }
    // end access ingress response routing
    
    // begin trunk ingress response routing

    // this could have been done as part of the first access port loop.
    // doing it separately to keep it simple.
    
    // note we can re-use the previously scanned valid and rlid arrays.    
    
    // each trunk port needs entries for GPUs on the near node only.
    // trunk ports do not see responses from far node GPUs.
    
    for ( portIndex = 0; portIndex < EXPLORER16_KT_NUM_GPU_PER_NODE; portIndex++ )
    { 
        for ( gpuIndex = 0; gpuIndex < EXPLORER16_KT_NUM_GPU_PER_NODE ; gpuIndex++ )
        {
        
            index = rlid[nodeIndex][gpuIndex];
            makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                     willowIndex,                               // willowIndex
                                     pTrunkPorts[portIndex],                    // portIndex
                                     index,                                     // Ingress req table index
                                     0,                                         // routePolicy
                                     valid7_0[nodeIndex][gpuIndex],             // vcModeValid7_0
                                     valid15_8[nodeIndex][gpuIndex],            // vcModeValid15_8, port 8 to 13
                                     valid17_16[nodeIndex][gpuIndex],           // vcModeValid17_16
                                     1);                                        // entryValid

        
        }
    }
 
    
}

void explorer16KTConfig::makeGangedLinkTable( int nodeIndex, int willowIndex ) 
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

void explorer16KTConfig::makeAccessPorts( int nodeIndex, int willowIndex ) 
{
    int i;
    int baseIndex = (nodeIndex * EXPLORER16_KT_NUM_SWITCH_PER_NODE * EXPLORER16_KT_NUM_GPU_PER_NODE) +
                    (willowIndex * EXPLORER16_KT_NUM_GPU_PER_NODE);

    for ( i = 0; i < EXPLORER16_KT_NUM_GPU_PER_NODE; i++ )   
    {
        makeOneAccessPort(nodeIndex,                            // nodeIndex 
                          willowIndex,                          // willowIndex
                          HWLinks16[baseIndex + i].willowPort,  // portIndex
                          0,                                    // farNodeID TODO Hmm farNodeId makes no sense for an access port
                          HWLinks16[baseIndex + i].GPUIndex,    // farPeerID         
                          HWLinks16[baseIndex + i].GPUPort,     // farPortNum        
                          DC_COUPLED);                          // portMode
    }
}

void explorer16KTConfig::makeTrunkPorts( int nodeIndex, int willowIndex ) 
{
    int i, farWillowID;

    if ( nodeIndex == 0 )
    {
        farWillowID = willowIndex + 0x18;
    }
    else
    {
        farWillowID = willowIndex + 0x08;
    }

    // For 16-GPU Explorer, physical ID equals board-relative index plus 0x08 on node 0.
    // Board-relative index plus 0x18 on node 1.


    for ( i = 0; i < EXPLORER16_KT_NUM_GPU_PER_NODE; i++ )
    {
        makeOneTrunkPort( nodeIndex, 
                          willowIndex,
                          trunkPortsNear[i],
                          nodeIndex ? 1 : 0,
                          farWillowID,
                          trunkPortsFar[i],
                          DC_COUPLED );
                          
    }
    return;
}

void explorer16KTConfig::makeOneWillow( int nodeIndex, int willowIndex ) 
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( (nodeIndex < EXPLORER16_KT_NUM_NODES) && (willowIndex >= 0) && (willowIndex < EXPLORER16_KT_NUM_SWITCH_PER_NODE)  )
    {
        switches[nodeIndex][willowIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][willowIndex]->set_ecid( ecid.str().c_str() );
        
        // For 16-GPU Explorer, physical ID equals board-relative index plus 0x08 on node 0.
        // Board-relative index plus 0x18 on node 1.
        if ( nodeIndex == 0 ) 
        {
            switches[nodeIndex][willowIndex]->set_physicalid( willowIndex + 0x08 );
        } 
        else
        {
            switches[nodeIndex][willowIndex]->set_physicalid( willowIndex + 0x18 );
        }    
        
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
        std::cout <<  "Invalid willowIndex" << willowIndex << "\n";
    }
}

void explorer16KTConfig::fillSystemPartitionInfo(int nodeIndex)
{
    node *nodeInfo = nodes[nodeIndex];

    nodeSystemPartitionInfo *systemPartInfo = new nodeSystemPartitionInfo();

    //TODO: setting this one for KT, create new partition in subsequent checkin 
    ptVMPartitionInfo *ptPartition = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata = new partitionMetaDataInfo();
    ptMetadata->set_gpucount( EXPLORER16_KT_NUM_GPU_PER_NODE );
    ptMetadata->set_switchcount( EXPLORER16_KT_NUM_SWITCH_PER_NODE );
    ptMetadata->set_lwlinkintratrunkconncount( 0 );
    ptMetadata->set_lwlinkintertrunkconncount( 48 );
    ptPartition->set_allocated_metadata( ptMetadata );

    nodes[nodeIndex]->set_allocated_partitioninfo( systemPartInfo );
}

void explorer16KTConfig::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
{
    int i, j;

    // Add Willows
    for ( i = 0; i < willowNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneWillow( nodeIndex, i);
    }

    fillSystemPartitionInfo( nodeIndex );
}
void explorer16KTConfig::makeNodes() 
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    //First allocate up all nodes
    for (nodeIndex = 0; nodeIndex < EXPLORER16_KT_NUM_NODES; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        if(nodeIndex == 0) {
            nodes[nodeIndex]->set_ipaddress( "192.168.122.245");
            nodes[nodeIndex]->set_nodeid( nodeIndex + 1);
        } else {
            nodes[nodeIndex]->set_ipaddress( "192.168.122.40");
            nodes[nodeIndex]->set_nodeid( nodeIndex + 1);
        }

    }

    //now set up GPUs for all nodes
    for (nodeIndex = 0; nodeIndex < EXPLORER16_KT_NUM_NODES; nodeIndex++)
    {
        for ( i = 0; i < EXPLORER16_KT_NUM_GPU_PER_NODE; i++ )
        {
            gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
            int endPointID = (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE) + i;
            makeOneGpu( nodeIndex, 
                        i, 
                        endPointID, 
                        0, 0x3F, 0xFFFFFFFF,
                        gpuFabricAddrBase[endPointID], 
                        gpuFabricAddrRange[endPointID], i + (nodeIndex * EXPLORER16_KT_NUM_GPU_PER_NODE));
        }

    }


    //now set up will and routing tables for all nodes
    for (nodeIndex = 0; nodeIndex < EXPLORER16_KT_NUM_NODES; nodeIndex++)
    {
        makeOneNode( nodeIndex, EXPLORER16_KT_NUM_GPU_PER_NODE, EXPLORER16_KT_NUM_SWITCH_PER_NODE);
    }
}
