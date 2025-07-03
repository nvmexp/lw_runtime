#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <string.h>
#include "explorer16common.h"
#include "explorer16Config.h"

explorer16Config::explorer16Config( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t i;
    for (i = 0; i < 16; i++) 
    {
        gpuFabricAddrBase[i]  = i << 36;
        gpuFabricAddrRange[i] = FAB_ADDR_RANGE_16G * 2;
    }
};

explorer16Config::~explorer16Config() 
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void explorer16Config::makeIngressReqTable( int nodeIndex, int willowIndex ) 
{
    int32_t gpuEndpointId, portIndex, portGPU, peerGPU, peerGPULink, gpuBase, gpuBaseFar;
    int32_t gpuIndex, linkBase, linkIndex, index, routeIndex;
    int64_t mappedAddr, range, i;
    GPU *gpu;
    uint32_t valid7_0[16]   = {0};  // vc_valid bits for each GPU 
    uint32_t valid15_8[16]  = {0};
    uint32_t valid17_16[16] = {0};
    linkBase = willowIndex * 8;
    gpuBase = 0;
    gpuBaseFar = 8;
    if ( willowIndex >= 6 )
    {
       gpuBase = 8;
       gpuBaseFar = 0;
    }

    // begin access ingress routing
    // first poplulate the vcValid array with correct egress for each access ingress on the local baseboard

    for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + 8; portIndex++ )
        { 
            //find the first port for the current GPU
            if ( HWLinks16[portIndex].GPUIndex == gpuIndex )
            {
                if (HWLinks16[portIndex].willowPort < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }
                break;
            } else {
                continue;
            }
        }
    }
    // for the far GPUs we can select the egress in order of appearance
    for ( gpuIndex = gpuBaseFar; gpuIndex < gpuBaseFar + 8; gpuIndex++ )
    {
        if (trunkPortsNear[gpuIndex] < 8)
        {
            valid7_0[gpuIndex] = 1 << (4 * trunkPortsNear[gpuIndex]);
        } 
        else if (trunkPortsNear[gpuIndex] < 16)
        {
            valid15_8[gpuIndex] = 1 << (4 * (trunkPortsNear[gpuIndex] - 8) );
        }
        else //these are not used for trunk links but we may want this code elsewhere so keeping it generic.
        {
            valid17_16[gpuIndex] = 1 << (4 * (trunkPortsNear[gpuIndex] - 16) );
        }
    }
             
    // now build the request table for each port for GPUs on this baseboard      
    for ( portIndex = 0; portIndex <= 7; portIndex++ )
    {
        for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
        {
            if ( HWLinks16[linkBase + portIndex].GPUIndex == gpuIndex )
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

                makeOneIngressReqEntry( nodeIndex,                          // nodeIndex
                                        willowIndex,                        // willowIndex
                                        HWLinks16[linkBase + portIndex].willowPort,
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
    
    // now build the request table for each port for GPUs on the far baseboard
    for ( portIndex = 0; portIndex <= 7; portIndex++ )
    {
        for ( gpuIndex = gpuBaseFar; gpuIndex < gpuBaseFar + 8; gpuIndex++ )
        {
            gpu = gpus[nodeIndex][gpuIndex];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();
            
            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                //Trunks have to preserve target address
                mappedAddr = gpu->fabricaddrbase() + (i << 34);

                makeOneIngressReqEntry( nodeIndex,                          // nodeIndex
                                        willowIndex,                        // willowIndex
                                        HWLinks16[linkBase + portIndex].willowPort,          
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
    // end access ingress request routing
    
    // begin trunk ingress request routing
    
    // this could have been done as part of the first access port loop.
    // doing it separately to keep it simple.
    
    // each trunk port needs entries for GPUs on the near baseboard only. 
    // build a list of which port goes to which GPU.
    for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + 8; portIndex++ )
        { 
            //find the first port for the current GPU
            if ( HWLinks16[portIndex].GPUIndex == gpuIndex )
            {
                if (HWLinks16[portIndex].willowPort < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }
                break;
            } else {
                continue;
            }
        }
    }
    
    // note that any trunk port on the board could get data for any GPU, so all
    // trunk port ingress tables are identical. 
    
    for ( portIndex = 0; portIndex <= 7; portIndex++ )
    {
        for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
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
                                        valid7_0[gpuIndex],                 // vcModeValid7_0
                                        valid15_8[gpuIndex],                // vcModeValid15_8, port 8 to 13
                                        valid17_16[gpuIndex],               // vcModeValid17_16
                                        1);                                 // entryValid
            }
        }
    }

};

void explorer16Config::makeIngressRespTable( int nodeIndex, int willowIndex ) 
{
    int gpuIndex, portIndex, i, index, enpointID, outPortNum, linkBase, linkBaseFar, gpuBase, gpuBaseFar, routeIndex;
    accessPort *outPort;
    unsigned int rlid[16];
    uint32_t valid7_0[16]   = {0};  // vc_valid bits for each GPU 
    uint32_t valid15_8[16]  = {0};
    uint32_t valid17_16[16] = {0};
    linkBase = willowIndex * 8;
    linkBaseFar = ( willowIndex + 6 ) * 8;
    gpuBase = 0;
    gpuBaseFar = 8;
    if (willowIndex >= 6) 
    {
        gpuBase = 8;
        gpuBaseFar = 0;
        linkBaseFar = ( willowIndex - 6 ) * 8;
    }

    // begin access ingress response routing
    // first poplulate the vcValid array with correct egress for each access ingress

    for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
    {
        for ( portIndex = linkBase; portIndex < linkBase + 8; portIndex++ )
        {
            if ( HWLinks16[portIndex].GPUIndex == gpuIndex )
            { 
                if (HWLinks16[portIndex].willowPort < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * HWLinks16[portIndex].willowPort);
                } 
                else if (HWLinks16[portIndex].willowPort < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 8) );
                }
                else 
                {
                    valid17_16[gpuIndex] = 1 << (4 * (HWLinks16[portIndex].willowPort - 16) );
                }

                // also record the egress, and requester link ID of the connecting port
                rlid[gpuIndex] = 6 * gpuIndex + HWLinks16[portIndex].GPUPort;
             
                break;
            } else {
                continue;
            }
        }
    }
    // for the far GPUs we can select the egress in order of appearance
    for ( gpuIndex = gpuBaseFar; gpuIndex < gpuBaseFar + 8; gpuIndex++ )
    {
        for ( portIndex = linkBaseFar; portIndex < linkBaseFar + 8; portIndex++ )
        {
            if ( HWLinks16[portIndex].GPUIndex == gpuIndex ) 
            {
                if (trunkPortsNear[gpuIndex] < 8)
                {
                    valid7_0[gpuIndex] = 1 << (4 * trunkPortsNear[gpuIndex]);
                } 
                else if (trunkPortsNear[gpuIndex] < 16)
                {
                    valid15_8[gpuIndex] = 1 << (4 * (trunkPortsNear[gpuIndex] - 8) );
                }
                else //these are not used for trunk links but we may want this code elsewhere so keeping it generic.
                {
                    valid17_16[gpuIndex] = 1 << (4 * (trunkPortsNear[gpuIndex] - 16) );
                }
                rlid[gpuIndex] = 6 * gpuIndex + HWLinks16[portIndex].GPUPort;
                //printf("far rlid %d, gpu %d, portIndex %d, port %d\n",rlid[gpuIndex],gpuIndex,portIndex,HWLinks16[portIndex].GPUPort);
            }
        }
    }
             
                
    for ( portIndex = 0; portIndex < 8; portIndex++ )
    { 
        for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
        {
        // each egress serves 1 ingress from each of the other GPUs
            if ( HWLinks16[linkBase + portIndex].GPUIndex == gpuIndex )
            { 
                continue;
            }
            else
            {
                index = rlid[gpuIndex];
 
                makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                         willowIndex,                               // willowIndex
                                         HWLinks16[linkBase + portIndex].willowPort,                          
                                                                                    // portIndex
                                         index,                                     // Ingress req table index
                                         0,                                         // routePolicy
                                         valid7_0[gpuIndex],                        // vcModeValid7_0
                                         valid15_8[gpuIndex],                       // vcModeValid15_8, port 8 to 13
                                         valid17_16[gpuIndex],                      // vcModeValid17_16
                                         1);                                        // entryValid

            }
        }
    }
    // now build the response table for each port for GPUs on the far baseboard
    for ( portIndex = 0; portIndex <= 7; portIndex++ )
    {
        for ( gpuIndex = gpuBaseFar; gpuIndex < gpuBaseFar + 8; gpuIndex++ )
        {
            index = rlid[gpuIndex];
            //printf("willow %d gpu %d index %d\n",willowIndex,gpuIndex,index);
            makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                     willowIndex,                               // willowIndex
                                     HWLinks16[linkBase + portIndex].willowPort,                          
                                                                                // portIndex
                                     index,                                     // Ingress req table index
                                     0,                                         // routePolicy
                                     valid7_0[gpuIndex],                        // vcModeValid7_0
                                     valid15_8[gpuIndex],                       // vcModeValid15_8, port 8 to 13
                                     valid17_16[gpuIndex],                      // vcModeValid17_16
                                     1);                                        // entryValid
        
        }
    }
    // end access ingress response routing
    
    // begin trunk ingress response routing

    // this could have been done as part of the first access port loop.
    // doing it separately to keep it simple.
    
    // note we can re-use the previously scanned valid and rlid arrays.    
    
    // each trunk port needs entries for GPUs on the near baseboard only.
    // trunk ports do not see responses from far baseboard GPUs.
    
    //printf("start trunk\n");
    for ( portIndex = 0; portIndex < 8; portIndex++ )
    { 
        for ( gpuIndex = gpuBase; gpuIndex < gpuBase + 8; gpuIndex++ )
        {
        
            index = rlid[gpuIndex];
            //printf("willow %d, port %d, index %d, valid %x %x %x\n",willowIndex, trunkPortsNear[portIndex],index,valid7_0[gpuIndex],valid15_8[gpuIndex],valid17_16[gpuIndex]);
            makeOneIngressRespEntry( nodeIndex,                                 // nodeIndex
                                     willowIndex,                               // willowIndex
                                     trunkPortsNear[portIndex],                 // portIndex
                                     index,                                     // Ingress req table index
                                     0,                                         // routePolicy
                                     valid7_0[gpuIndex],                        // vcModeValid7_0
                                     valid15_8[gpuIndex],                       // vcModeValid15_8, port 8 to 13
                                     valid17_16[gpuIndex],                      // vcModeValid17_16
                                     1);                                        // entryValid

        
        }
    }
 
    
}

void explorer16Config::makeGangedLinkTable( int nodeIndex, int willowIndex ) 
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

void explorer16Config::makeAccessPorts( int nodeIndex, int willowIndex ) 
{
    int i;
    int baseIndex = willowIndex * 8;

    for ( i = 0; i < 8; i++ )   
    {
        makeOneAccessPort(0,                                    // nodeIndex 
                          willowIndex,                          // willowIndex
                          HWLinks16[baseIndex + i].willowPort,    // portIndex
                          0,                                    // farNodeID        
                          HWLinks16[baseIndex + i].GPUIndex,      // farPeerID         
                          HWLinks16[baseIndex + i].GPUPort,       // farPortNum        
                          DC_COUPLED);                          // portMode
    }
}

void explorer16Config::makeTrunkPorts( int nodeIndex, int willowIndex ) 
{
    int i, farWillowID;

    if ( willowIndex < 6 )
    {
        farWillowID = willowIndex + 6 - 0x06 + 0x18;
    }
    else
    {
        farWillowID = willowIndex - 6 + 0x08;
    }

    // For 16-GPU Explorer, physical ID equals board-relative index plus 0x08 on baseboard 0.
    // Board-relative index plus 0x18 on baseboard 1.


    for ( i = 0; i < 8; i++ )
    {
        makeOneTrunkPort( nodeIndex, 
                          willowIndex,
                          trunkPortsNear[i],
                          0,
                          farWillowID,
                          trunkPortsFar[i],
                          DC_COUPLED );
                          
    }
    return;
}

void explorer16Config::makeOneWillow( int nodeIndex, int willowIndex ) 
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( ( (nodeIndex == 0) && (willowIndex >= 0) ) &&
         ( (nodeIndex == 0) && (willowIndex <= 11) ) )
    {
        switches[nodeIndex][willowIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][willowIndex]->set_ecid( ecid.str().c_str() );
        
        // For 16-GPU Explorer, physical ID equals board-relative index plus 0x08 on baseboard 0.
        // Board-relative index plus 0x18 on baseboard 1.
        if ( willowIndex < 6 ) 
        {
            switches[nodeIndex][willowIndex]->set_physicalid( willowIndex + 0x08 );
        } 
        else
        {
            switches[nodeIndex][willowIndex]->set_physicalid( willowIndex - 0x06 + 0x18 );
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
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void explorer16Config::fillSystemPartitionInfo(int nodeIndex)
{
    node *nodeInfo = nodes[nodeIndex];

    nodeSystemPartitionInfo *systemPartInfo = new nodeSystemPartitionInfo();

    // fill all the bare metal partition information

    // DGX-2 (explorer16 ) bare metal partition information
    bareMetalPartitionInfo *bareMetalPartInfo1 = systemPartInfo->add_baremetalinfo();
    partitionMetaDataInfo *bareMetaData1 = new partitionMetaDataInfo();
    bareMetaData1->set_gpucount( nodeInfo->gpu_size() );
    bareMetaData1->set_switchcount( nodeInfo->lwswitch_size() );
    // total interanode trunk connections 48  (6switch * 8)
    bareMetaData1->set_lwlinkintratrunkconncount( 48 );
    // no internode trunk connection for explorer16
    bareMetaData1->set_lwlinkintertrunkconncount( 0 );
    bareMetalPartInfo1->set_allocated_metadata( bareMetaData1 );

    // HGX-2 multi-host system bare metal partition information
    bareMetalPartitionInfo *bareMetalPartInfo2 = systemPartInfo->add_baremetalinfo();
    partitionMetaDataInfo *bareMetaData2 = new partitionMetaDataInfo();
    bareMetaData2->set_gpucount( 8 );
    bareMetaData2->set_switchcount( 6 );
    // no interanode trunk connections (baseboards are not connected)
    bareMetaData2->set_lwlinkintratrunkconncount( 0 );
    // no internode trunk connection for explorer16
    bareMetaData2->set_lwlinkintertrunkconncount( 0 );
    bareMetalPartInfo2->set_allocated_metadata( bareMetaData2 );

    // fill all the Pass-through virtualization partition information

    //
    //  GPUs     Switches    Number of trunk connections
    //    16        12       48
    //    8         6        0
    //    4         3        0
    //    2         1        0
    //    1         0        0
    //
    ptVMPartitionInfo *ptPartition1 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata1 = new partitionMetaDataInfo();
    ptMetadata1->set_gpucount( 16 );
    ptMetadata1->set_switchcount( 12 );
    ptMetadata1->set_lwlinkintratrunkconncount( 48 );
    ptMetadata1->set_lwlinkintertrunkconncount( 0 );
    ptPartition1->set_allocated_metadata( ptMetadata1 );

    ptVMPartitionInfo *ptPartition2 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata2 = new partitionMetaDataInfo();
    ptMetadata2->set_gpucount( 8 );
    ptMetadata2->set_switchcount( 6 );
    ptMetadata2->set_lwlinkintratrunkconncount( 0 );
    ptMetadata2->set_lwlinkintertrunkconncount( 0 );
    ptPartition2->set_allocated_metadata( ptMetadata2 );

    ptVMPartitionInfo *ptPartition3 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata3 = new partitionMetaDataInfo();
    ptMetadata3->set_gpucount( 4 );
    ptMetadata3->set_switchcount( 3 );
    ptMetadata3->set_lwlinkintratrunkconncount( 0);
    ptMetadata3->set_lwlinkintertrunkconncount( 0 );
    ptPartition3->set_allocated_metadata( ptMetadata3 );

    ptVMPartitionInfo *ptPartition4 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata4 = new partitionMetaDataInfo();
    ptMetadata4->set_gpucount( 2 );
    ptMetadata4->set_switchcount( 1 );
    ptMetadata4->set_lwlinkintratrunkconncount( 0 );
    ptMetadata4->set_lwlinkintertrunkconncount( 0 );
    ptPartition4->set_allocated_metadata( ptMetadata4 );

    ptVMPartitionInfo *ptPartition5 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata5 = new partitionMetaDataInfo();
    ptMetadata5->set_gpucount( 1 );
    ptMetadata5->set_switchcount( 0 );
    ptMetadata5->set_lwlinkintratrunkconncount( 0 );
    ptMetadata5->set_lwlinkintertrunkconncount( 0 );
    ptPartition5->set_allocated_metadata( ptMetadata5 );

  
    // fill all the GPU Pass-through only (Shared LWSwitch) virtualization partition information
    for ( uint32 idx = 0; idx < MAX_SHARED_LWSWITCH_FABRIC_PARTITIONS; idx++ )
    {
        SharedPartInfoTable_t partEntry = gSharedVMPartInfo[idx];
        sharedLWSwitchPartitionInfo *sharedPartition = systemPartInfo->add_sharedlwswitchinfo();
        partitionMetaDataInfo *sharedMetaData = new partitionMetaDataInfo();
        sharedMetaData->set_gpucount( partEntry.gpuCount );
        sharedMetaData->set_switchcount( partEntry.switchCount );
        sharedMetaData->set_lwlinkintratrunkconncount( partEntry.lwLinkIntraTrunkConnCount );
        sharedMetaData->set_lwlinkintertrunkconncount( partEntry.lwLinkInterTrunkConnCount );

        sharedPartition->set_partitionid( partEntry.partitionId );
        sharedPartition->set_allocated_metadata( sharedMetaData );
        // populate all the GPU information
        for ( uint32 gpuIdx = 0; gpuIdx < partEntry.gpuCount; gpuIdx++ )
        {
            SharedPartGpuInfoTable_t gpuEntry = partEntry.gpuInfo[gpuIdx];
            sharedLWSwitchPartitionGpuInfo *partGpu = sharedPartition->add_gpuinfo();
            partGpu->set_physicalid( gpuEntry.physicalId );
            partGpu->set_numenabledlinks( gpuEntry.numEnabledLinks );
            partGpu->set_enabledlinkmask ( gpuEntry.enabledLinkMask );
        }

        // populate all the Switch information
        for ( uint32 switchIdx = 0; switchIdx < partEntry.switchCount; switchIdx++ )
        {
            SharedPartSwitchInfoTable_t switchEntry = partEntry.switchInfo[switchIdx];
            sharedLWSwitchPartitionSwitchInfo *partSwitch = sharedPartition->add_switchinfo();
            partSwitch->set_physicalid( switchEntry.physicalId );
            partSwitch->set_numenabledlinks( switchEntry.numEnabledLinks );
            partSwitch->set_enabledlinkmask ( switchEntry.enabledLinkMask );
        }
    }

    nodes[nodeIndex]->set_allocated_partitioninfo( systemPartInfo );
}

void explorer16Config::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

    fillSystemPartitionInfo( nodeIndex );
}

void explorer16Config::makeNodes() 
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
            // node 0 has 16 GPUs, 12 Willows
            makeOneNode( nodeIndex, 16, 12);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
explorer16Config::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
explorer16Config::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorer16Config::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
explorer16Config::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
