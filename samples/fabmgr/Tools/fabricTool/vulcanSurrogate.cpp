#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>
#include "json.h"

#include "fabricConfig.h"
#include "vulcanSurrogate.h"

//
// This is a vulcan surrogate board topology with 8 GPUs and 4 LWSwitches
// LWLink connection details and LWSwitch GPIO physical IDs are in
// //syseng/Projects/DGX/Boards/P5612_Vulcan_GPU_Base_Board/Docs/P5612_Lwlink_Mapping_sheet_A00.xlsx#2
// https://lwbugs/3324387
//

vulcanSurrogate::vulcanSurrogate( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t targetId;

    mNumSwitches = 4;
    mNumGpus = 8;

    for ( targetId = 0; targetId < mNumGpus; targetId++ )
    {
        gpuTargetId[targetId] = targetId;
        gpuFabricAddrBase[targetId]  = (targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE) << 36;
        gpuFabricAddrRange[targetId] = FAB_ADDR_RANGE_16G * 8;

        gpuFlaAddrBase[targetId]  = (uint64_t)(targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + LR_FIRST_FLA_RMAP_SLOT) << 36;
        gpuFlaAddrRange[targetId] = FAB_ADDR_RANGE_16G * 8;
    }

    mEnableTrunkLookback  = false;
    mEnableSpray   = false;
    mMaxTargetId = 0;

    mSharedPartitionJsonFile = NULL;
    mSharedPartitiolwersion = 0;
    mNumSharedPartitions = VULCAN_SURROGATE_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS;

    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mSharedPartitionTimeStamp = asctime(loc_time);

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );
};

vulcanSurrogate::~vulcanSurrogate()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void
vulcanSurrogate::setConfig( bool enableTrunkLookback, bool enableSpray, const char *sharedPartitionJsonFile )
{
    uint32_t swIndex, portIndex;
    const uint32_t numSwitchesPerBaseboard = 4;

    mEnableTrunkLookback = enableTrunkLookback;
    mEnableSpray   = enableSpray;

    mNumSwitches = 4;
    mNumGpus = 8;

    if ( sharedPartitionJsonFile != NULL )
    {
        mSharedPartitionJsonFile = sharedPartitionJsonFile;
    }

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );

    // Access ports
    // Switch 0
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][10]  = {0,      0,            0,      0,        true,         0,                11,           false,       0,            0,                0};
    mSwitchPorts[0][11]  = {0,      0,            0,      1,        true,         0,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][14]  = {0,      0,            0,      0,        true,         1,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][15]  = {0,      0,            0,      1,        true,         1,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][25]  = {0,      0,            0,      0,        true,         2,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][24]  = {0,      0,            0,      1,        true,         2,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][8]   = {0,      0,            0,      0,        true,         3,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][9]   = {0,      0,            0,      1,        true,         3,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][31]  = {0,      0,            0,      0,        true,         4,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][30]  = {0,      0,            0,      1,        true,         4,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][12] =  {0,      0,            0,      0,        true,         5,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][13] =  {0,      0,            0,      1,        true,         5,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][29] =  {0,      0,            0,      0,        true,         6,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][28] =  {0,      0,            0,      1,        true,         6,                 0,           false,       0,            0,                0};
    mSwitchPorts[0][27] =  {0,      0,            0,      0,        true,         7,                 0,           false,       0,            0,                0};
    mSwitchPorts[0][26] =  {0,      0,            0,      1,        true,         7,                 2,           false,       0,            0,                0};

    // Switch 1
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][9]   = {0,      0,            0,      0,        true,         0,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][15]  = {0,      0,            0,      1,        true,         0,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][4]   = {0,      0,            0,      0,        true,         1,                11,           false,       0,            0,                0};
    mSwitchPorts[1][1]   = {0,      0,            0,      1,        true,         1,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][13]  = {0,      0,            0,      0,        true,         2,                11,           false,       0,            0,                0};
    mSwitchPorts[1][0]   = {0,      0,            0,      1,        true,         2,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][31]  = {0,      0,            0,      0,        true,         3,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][28]  = {0,      0,            0,      1,        true,         3,                11,           false,       0,            0,                0};
    mSwitchPorts[1][26]  = {0,      0,            0,      0,        true,         4,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][24]  = {0,      0,            0,      1,        true,         4,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][12]  = {0,      0,            0,      0,        true,         5,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][5]   = {0,      0,            0,      1,        true,         5,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][30]  = {0,      0,            0,      0,        true,         6,                10,           false,       0,            0,                0};
    mSwitchPorts[1][29]  = {0,      0,            0,      1,        true,         6,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][11]  = {0,      0,            0,      0,        true,         7,                10,           false,       0,            0,                0};
    mSwitchPorts[1][27]  = {0,      0,            0,      1,        true,         7,                 4,           false,       0,            0,                0};

    // Switch 2
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[2][14]  = {0,      0,            0,      0,        true,         0,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][31]  = {0,      0,            0,      1,        true,         0,                 9,           false,       0,            0,                0};
    mSwitchPorts[2][10]  = {0,      0,            0,      0,        true,         1,                10,           false,       0,            0,                0};
    mSwitchPorts[2][15]  = {0,      0,            0,      1,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][28]  = {0,      0,            0,      0,        true,         2,                 8,           false,       0,            0,                0};
    mSwitchPorts[2][16]  = {0,      0,            0,      1,        true,         2,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][13]  = {0,      0,            0,      0,        true,         3,                 0,           false,       0,            0,                0};
    mSwitchPorts[2][4]   = {0,      0,            0,      1,        true,         3,                 9,           false,       0,            0,                0};
    mSwitchPorts[2][9]   = {0,      0,            0,      0,        true,         4,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][27]  = {0,      0,            0,      1,        true,         4,                11,           false,       0,            0,                0};
    mSwitchPorts[2][8]   = {0,      0,            0,      0,        true,         5,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][0]   = {0,      0,            0,      1,        true,         5,                10,           false,       0,            0,                0};
    mSwitchPorts[2][26]  = {0,      0,            0,      0,        true,         6,                 8,           false,       0,            0,                0};
    mSwitchPorts[2][24]  = {0,      0,            0,      1,        true,         6,                 1,           false,       0,            0,                0};
    mSwitchPorts[2][30]  = {0,      0,            0,      0,        true,         7,                 9,           false,       0,            0,                0};
    mSwitchPorts[2][29]  = {0,      0,            0,      1,        true,         7,                 1,           false,       0,            0,                0};

    // Switch 3
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[3][31]  = {0,      0,            0,      0,        true,         0,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][30]  = {0,      0,            0,      1,        true,         0,                10,           false,       0,            0,                0};
    mSwitchPorts[3][12]  = {0,      0,            0,      0,        true,         1,                 2,           false,       0,            0,                0};
    mSwitchPorts[3][13]  = {0,      0,            0,      1,        true,         1,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][27]  = {0,      0,            0,      0,        true,         2,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][26]  = {0,      0,            0,      1,        true,         2,                10,           false,       0,            0,                0};
    mSwitchPorts[3][14]  = {0,      0,            0,      0,        true,         3,                10,           false,       0,            0,                0};
    mSwitchPorts[3][15]  = {0,      0,            0,      1,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[3][25]  = {0,      0,            0,      0,        true,         4,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][24]  = {0,      0,            0,      1,        true,         4,                10,           false,       0,            0,                0};
    mSwitchPorts[3][8]   = {0,      0,            0,      0,        true,         5,                 2,           false,       0,            0,                0};
    mSwitchPorts[3][9]   = {0,      0,            0,      1,        true,         5,                11,           false,       0,            0,                0};
    mSwitchPorts[3][10]  = {0,      0,            0,      0,        true,         6,                 9,           false,       0,            0,                0};
    mSwitchPorts[3][11]  = {0,      0,            0,      1,        true,         6,                11,           false,       0,            0,                0};
    mSwitchPorts[3][29]  = {0,      0,            0,      0,        true,         7,                 8,           false,       0,            0,                0};
    mSwitchPorts[3][28]  = {0,      0,            0,      1,        true,         7,                11,           false,       0,            0,                0};

    // only define trunk ports when enableTrunkLookback
    // MODS does not like unused trunk ports
    if ( enableTrunkLookback )
    {
        // Switch 0
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[0][2]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            0,                6};
        mSwitchPorts[0][3]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            0,                7};
        mSwitchPorts[0][6]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            0,                2};
        mSwitchPorts[0][7]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            0,                3};
        mSwitchPorts[0][18]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            0,               22};
        mSwitchPorts[0][19]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            0,               23};
        mSwitchPorts[0][22]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            0,               18};
        mSwitchPorts[0][23]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            0,               19};

        // Switch 1
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[1][2]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            1,                6};
        mSwitchPorts[1][3]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            1,                7};
        mSwitchPorts[1][6]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            1,                2};
        mSwitchPorts[1][7]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            1,                3};
        mSwitchPorts[1][17]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            1,               21};
        mSwitchPorts[1][18]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            1,               22};
        mSwitchPorts[1][19]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            1,               23};
        mSwitchPorts[1][21]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            1,               17};
        mSwitchPorts[1][22]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            1,               18};
        mSwitchPorts[1][23]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            1,               19};

        // Switch 2
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[2][2]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            2,                6};
        mSwitchPorts[2][3]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            2,                7};
        mSwitchPorts[2][6]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            2,                2};
        mSwitchPorts[2][7]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            2,                3};
        mSwitchPorts[2][17]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            2,               21};
        mSwitchPorts[2][18]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            2,               22};
        mSwitchPorts[2][19]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            2,               23};
        mSwitchPorts[2][21]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            2,               17};
        mSwitchPorts[2][22]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            2,               18};
        mSwitchPorts[2][23]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            2,               19};

        // Switch 3
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[3][2]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            3,                6};
        mSwitchPorts[3][3]   = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            3,                7};
        mSwitchPorts[3][6]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            3,                2};
        mSwitchPorts[3][7]   = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            3,                3};
        mSwitchPorts[3][18]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            3,               22};
        mSwitchPorts[3][19]  = {0,      0,            0,      0,       false,         0,                 0,            true,       0,            3,               23};
        mSwitchPorts[3][22]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            3,               18};
        mSwitchPorts[3][23]  = {0,      0,            0,      1,       false,         0,                 0,            true,       0,            3,               19};
    }

    // Switch ports setting
    for ( swIndex = 0; swIndex < mNumSwitches; swIndex++ )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            mSwitchPorts[swIndex][portIndex].nodeId = 0;

            //
            // Switch GPIO:
            // according to http://lwbugs/3324387/10
            // Switch GPIO physical ID is the same as 0 based swIndex
            //
            mSwitchPorts[swIndex][portIndex].swPhysicalId = swIndex;

            // port index
            mSwitchPorts[swIndex][portIndex].swPort = portIndex;

        } // portIndex loop
    } // swIndex loop

    printf("index nodeId swPhysicalId swPort connectToGpu peerGpuPhysicalId peerGpuPort connectToSw peerSwNodeId peerSwPhysicalId peerSwPort\n");
    for ( int j = 0; j < 2; j++ )
    {
        printf("Switch %d\n", j);

        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            printf("%5d %6d %12d %6d %12d %17d %11d %11d %12d %16d %10d\n",
                    i, mSwitchPorts[j][i].nodeId, mSwitchPorts[j][i].swPhysicalId, mSwitchPorts[j][i].swPort, mSwitchPorts[j][i].connectToGpu,
                    mSwitchPorts[j][i].peerGpuPhysicalId, mSwitchPorts[j][i].peerGpuPort, mSwitchPorts[j][i].connectToSw, mSwitchPorts[j][i].peerSwNodeId,
                    mSwitchPorts[j][i].peerSwPhysicalId, mSwitchPorts[j][i].peerSwPort);
        }
    }
}

void vulcanSurrogate::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t portIndex;
    int64_t mappedAddr, range, i, index;
    uint32_t gpuPhysicalId, targetId;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        // on all access ports
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[swIndex][portIndex].connectToGpu == false )
                continue;

            for ( gpuPhysicalId = 0; gpuPhysicalId < mNumGpus; gpuPhysicalId++ )
            {
                targetId = gpuTargetId[gpuPhysicalId];

                for ( i = 0; i < NUM_INGR_RMAP_ENTRIES_PER_AMPERE; i++ )
                {
                    // GPA map slot
                    index = targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + i;
                    range = gpuFabricAddrRange[gpuPhysicalId];
                    mappedAddr = (index % NUM_INGR_RMAP_ENTRIES_PER_AMPERE) << 36; // increase each slot by 64G

                    makeOneRemapEntry( nodeIndex,           // nodeIndex
                                       swIndex,             // swIndex
                                       portIndex,           // portIndex
                                       index,               // rmap table index
                                       1,                   // entry valid
                                       mappedAddr,          // 64 bits remap fabric address (remap to 0 physical)
                                       0,                   // context match
                                       0,                   // context mask
                                       0,                   // context replacement
                                       0,                   // address offset
                                       0,                   // address base
                                       0,                   // address limit
                                       targetId,            // target ID
                                       1,                   // rfunc = remap address. skip context and range checks
                                       0,                   // irl select
                                       0,                   // FAM swizzle
                                       0,                   // FAM mult2
                                       0);                  // FAM plane select

                    // FLA map slot
                    index = LR_FIRST_FLA_RMAP_SLOT + targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + i;
                    range = gpuFlaAddrRange[gpuPhysicalId];
                    mappedAddr = index << 36; // increase each slot by 64G

                    makeOneRemapEntry( nodeIndex,           // nodeIndex
                                       swIndex,             // swIndex
                                       portIndex,           // portIndex
                                       index,               // rmap table index
                                       1,                   // entry valid
                                       mappedAddr,          // 64 bits remap FLA to itsef, due to bug 2498189
                                       0,                   // context match
                                       0,                   // context mask
                                       0,                   // context replacement
                                       0,                   // address offset
                                       0,                   // address base
                                       0,                   // address limit
                                       targetId,            // target ID
                                       1,                   // remap FLA address. skip context and range checks
                                       0,                   // irl select
                                       0,                   // FAM swizzle
                                       0,                   // FAM mult2
                                       0);                  // FAM plane select
                }
            }
        }
    }
}

void vulcanSurrogate::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void vulcanSurrogate::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

int
vulcanSurrogate::getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex )
{
    int count = 0;
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToSw == true ) )
        {
            count++;
        }
    }
    return count;
}

int
vulcanSurrogate::getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPort )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].swPort == swPort ) )
        {
            return i;
        }
    }

    return -1;
}

int
vulcanSurrogate::getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId)
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToGpu == true ) &&
             ( mSwitchPorts[swIndex][i].peerGpuPort == rlanId ) )
        {
            return i;
        }
    }

    return -1;
}

bool
vulcanSurrogate::isGpuConnectedToSwitch( uint32_t swNodeId, uint32_t swIndex, uint32_t gpuNodeId, uint32_t gpuPhysicalId )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == swNodeId ) &&
             ( mSwitchPorts[swIndex][i].connectToGpu == true ) &&
             ( mSwitchPorts[swIndex][i].nodeId == gpuNodeId ) &&
             ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == gpuPhysicalId ) )
        {
            return true;
        }
    }

    return false;
}

void vulcanSurrogate::getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                       uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                       uint32_t *egressPortIndex, uint32_t *numEgressPorts, bool *isLastHop )
{
    int32_t i, portCount = 0;
    bool useTrunkPorts;

    *numEgressPorts = 0;
    *isLastHop = false;

    if ( isGpuConnectedToSwitch( nodeIndex, swIndex, destGpuNodeId, destGpuPhysicalId ) == true )
    {
        // destination GPU is on the same switch
        // could use trunk port to loopback
        useTrunkPorts = mEnableTrunkLookback;
    }
    else
    {
        // destination GPU is not on the same switch, use trunk ports
        useTrunkPorts = true;
    }

    int ingressPortIndex = getSwitchPortIndex(nodeIndex, swIndex, ingressPortNum);
    if ( ( ingressPortIndex < 0 ) || ( ingressPortIndex > MAX_PORTS_PER_LWSWITCH ) )
    {
        // cannot find the ingress port
        printf("%s: Cannot find the ingress port swNodeId %d, swPhysicalId %d, ingressPortNum %d.\n",
                __FUNCTION__, nodeIndex, swIndex, ingressPortNum);
        return;
    }

    SwitchPort_t ingressPort = mSwitchPorts[swIndex][ingressPortIndex];

    if ( ingressPort.connectToGpu )
    {
        // ingress port is an access port

        // destGpu is connected to the same ingress port
        if ( ( ingressPort.nodeId == destGpuNodeId ) &&
             ( ingressPort.peerGpuPhysicalId == destGpuPhysicalId ) )
        {
            // Do not go back the same ingress access port
            *numEgressPorts = 0;
            *isLastHop = true;
            return;
        }

        // destGpu is connected to access ports that are different from the ingress port
        for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( useTrunkPorts )
            {
                // useTrunkPorts is true, go to trunk ports
                if ( mSwitchPorts[swIndex][i].connectToSw == true )
                {
                    // TODO trunk loopback
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
            else
            {
                // useTrunkPorts is false, go to access ports connected to destGpu
                if ( ( mSwitchPorts[swIndex][i].connectToGpu ) &&
                     ( mSwitchPorts[swIndex][i].nodeId == destGpuNodeId ) &&
                     ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == destGpuPhysicalId ) )
                {
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
        }
        *numEgressPorts = portCount > INGRESS_RID_MAX_PORTS ? INGRESS_RID_MAX_PORTS : portCount;
        *isLastHop = (useTrunkPorts == true) ? false : true;
        return;
    }

    // ingress port is a trunk port
    if ( ingressPort.connectToSw )
    {
        // go to access ports connected to destGpu
        for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( ( mSwitchPorts[swIndex][i].connectToGpu ) &&
                 ( mSwitchPorts[swIndex][i].nodeId == destGpuNodeId ) &&
                 ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == destGpuPhysicalId ) )
            {
                egressPortIndex[portCount] = i;
                portCount++;
            }

        }
        *numEgressPorts = portCount > INGRESS_RID_MAX_PORTS ? INGRESS_RID_MAX_PORTS : portCount;
        *isLastHop = true;
        return;
    }
}

void vulcanSurrogate::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
{
    uint32_t i, egressPortCount;
    uint32_t egressPortIndex[INGRESS_RID_MAX_PORTS];
    int egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS];
    int groupSelect[INGRESS_RLAN_MAX_GROUPS], groupSize[INGRESS_RLAN_MAX_GROUPS];
    bool isLastHop;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( uint32_t gpuPhysicalId = 0; gpuPhysicalId < mNumGpus; gpuPhysicalId++ )
        {
            for ( uint32_t portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
            {
                int ingressPortIndex = getSwitchPortIndex(nodeIndex, swIndex, portIndex);
                if ( ( ingressPortIndex < 0 ) || ( ingressPortIndex > MAX_PORTS_PER_LWSWITCH ) )
                {
                    // cannot find the ingress port
                    printf("%s: Cannot find the ingress port swNodeId %d, swPhysicalId %d, ingressPortNum %d.\n",
                           __FUNCTION__, nodeIndex, swIndex, portIndex);
                    continue;
                }

                if ( ( mSwitchPorts[swIndex][ingressPortIndex].connectToGpu == false ) &&
                     ( mSwitchPorts[swIndex][ingressPortIndex].connectToSw == false ) )

                {
                    // the port is not connected to anything
                    continue;
                }

                memset(egressPort, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
                memset(vcMap, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
                memset(groupSelect, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);
                memset(groupSize, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);

                getEgressPortsToGpu( nodeIndex, swIndex, portIndex,
                                     nodeIndex, gpuPhysicalId,
                                     egressPortIndex, &egressPortCount, &isLastHop);

                if ( egressPortCount == 0 )
                {
                    printf("%s: there is no egress port to gpuPhysicalId %d from nodeIndex %d swIndex %d, portIndex %d.\n",
                           __FUNCTION__, gpuPhysicalId, nodeIndex, swIndex, portIndex);
                    continue;
                }
                else
                {
                    printf("%s: On port %d egress ports to gpuPhysicalId %d: ",
                            __FUNCTION__, portIndex, gpuPhysicalId);
                    for ( i = 0; i < egressPortCount; i++ )
                    {
                        printf("%d ", egressPortIndex[i]);
                    }
                    printf("\n");
                }

                if ( mEnableSpray == false )
                {
                    // no spray
                    uint32_t selectedIndex = 0xFFFF;

                    // ingress port is an access port
                    if ( mSwitchPorts[swIndex][ingressPortIndex].connectToGpu == true )
                    {
                        // egress port is an access port
                       if ( mSwitchPorts[swIndex][egressPortIndex[0]].connectToGpu == true )
                       {
                           // egress port is an access port
                           // To select one egress access port from a set of 2 access ports
                           // select the one with the same rlanId
                           // select only one port with the same rlanId from the egress port list
                           uint32_t j;
                           for ( j = 0; j < egressPortCount; j++ )
                           {
                               if ( mSwitchPorts[swIndex][egressPortIndex[j]].rlanId == mSwitchPorts[swIndex][ingressPortIndex].rlanId )
                               {
                                   selectedIndex = j;
                                   break;
                               }
                           }

                           if ( j >= egressPortCount )
                           {
                               printf("Failed to find an egress port with rlandId %d\n",
                                       mSwitchPorts[swIndex][ingressPortIndex].rlanId);
                               continue;
                           }
                       }
                       else
                       {
                           // egress port is a trunk port
                           // To select one egress trunk port from a set of 8 (Switch 0 and 3) or 10 (Switch 1 and 2) trunk ports
                           uint32_t srcGpuPhysicalId = mSwitchPorts[swIndex][ingressPortIndex].peerGpuPhysicalId;

                           // each GPU is connected to 2 access ports, rlanId 0 and rlanId 1
                           // use the ingress port to select an egress port
                           selectedIndex = (srcGpuPhysicalId*2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId) % egressPortCount;

                           if ( selectedIndex >= egressPortCount )
                           {
                               printf("selectedIndex %d is larger than egressPortCount %d.\n", selectedIndex, egressPortCount);
                               continue;
                           }
                       }
                    }
                    else
                    {
                        // ingress port is a trunk port
                        if ( mSwitchPorts[swIndex][egressPortIndex[0]].connectToGpu == true )
                        {
                            // egress port is an access port
                            // To select one egress access port from a set of 2 access ports
                            // select the one with the opposite rlanId
                            uint32_t j;
                            for ( j = 0; j < egressPortCount; j++ )
                            {
                                if ( mSwitchPorts[swIndex][egressPortIndex[j]].rlanId != mSwitchPorts[swIndex][ingressPortIndex].rlanId )
                                {
                                    selectedIndex = j;
                                    break;
                                }
                            }

                            if ( j >= egressPortCount )
                            {
                                printf("Failed to find an egress port with rlandId %d\n",
                                        mSwitchPorts[swIndex][ingressPortIndex].rlanId);
                                continue;
                            }
                        }
                        else
                        {
                            // egress port is a trunk port
                            // TODO for multinode
                        }
                    }

                    if ( selectedIndex > INGRESS_RID_MAX_PORTS )
                    {
                        printf("Failed to select an egress port to GPU gpuPhysicalId %d.\n", gpuPhysicalId);
                        continue;
                    }

                    vcMap[0] = 0;
                    egressPort[0] = mSwitchPorts[swIndex][egressPortIndex[selectedIndex]].swPort;
                    printf("select egress port selectedIndex %d, number %d\n",
                            selectedIndex, egressPort[0]);

                    // RID entry
                    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                          swIndex,          // swIndex
                                          portIndex,        // portIndex
                                          gpuTargetId[gpuPhysicalId],    // RID Route table index
                                          1,                // entry valid
                                          0,                // rmod (no special routing)
                                          1,                // number of ports
                                          vcMap,            // pointer to array of VC controls
                                          egressPort);      // pointer to array of ports

                    // RLAN entry
                    makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                           swIndex,          // swIndex
                                           portIndex,        // portIndex
                                           gpuTargetId[gpuPhysicalId],    // RID Route table index
                                           1,                // entry valid
                                           0,                // group count
                                           NULL,             // group select array
                                           NULL );           // group size array
                }
                else
                {
                    // spray
                    for ( uint32_t j = 0; j < egressPortCount; j++ )
                    {
                        groupSelect[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = j;
                        groupSize[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = 1;
                        egressPort[j] = mSwitchPorts[swIndex][egressPortIndex[j]].swPort;
                    }

                    int rmod = isLastHop ? 0x40 : 0;

                    // RID entry
                    makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                          swIndex,          // swIndex
                                          portIndex,        // portIndex
                                          gpuTargetId[gpuPhysicalId], // RID Route table index
                                          1,                // entry valid
                                          rmod,             // rmod (no special routing)
                                          egressPortCount,  // spray to all egress ports
                                          vcMap,            // pointer to array of VC controls
                                          egressPort);      // pointer to array of ports

                    // RLAN entry
                    makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                           swIndex,          // swIndex
                                           portIndex,        // portIndex
                                           gpuTargetId[gpuPhysicalId], // RLAN Route table index
                                           1,                // entry valid
                                           INGRESS_RLAN_MAX_GROUPS, // group count
                                           groupSelect,      // group select array
                                           groupSize );      // group size array

                }

            } // for portIndex
        } // gpuPhysicalId
    }
}

void vulcanSurrogate::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void vulcanSurrogate::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToGpu == false )
                continue;

            makeOneAccessPort( nodeIndex,
                               swIndex,
                               mSwitchPorts[swIndex][i].swPort,   // portIndex
                               0,                                 // farNodeID
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerID
                               mSwitchPorts[swIndex][i].peerGpuPort,       // farPortNum
                               DC_COUPLED,                                 // phyMode
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerTargetID
                               mSwitchPorts[swIndex][i].rlanId,   // rlanID, set to connected GPU port number
                               mMaxTargetId);                     // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void vulcanSurrogate::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToSw == false )
                continue;

            makeOneTrunkPort( nodeIndex,
                              swIndex,
                              mSwitchPorts[swIndex][i].swPort,  // portIndex
                              0,                                // farNodeID
                              mSwitchPorts[swIndex][i].peerSwPhysicalId, // farSwitchID
                              mSwitchPorts[swIndex][i].peerSwPort,       // farPortNum
                              DC_COUPLED,                       // phyMode
                              mMaxTargetId);                    // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void vulcanSurrogate::makeOneLwswitch( int nodeIndex, int swIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << swIndex;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        switches[nodeIndex][swIndex]->set_version( FABRIC_MANAGER_VERSION );
        switches[nodeIndex][swIndex]->set_ecid( ecid.str().c_str() );

        // Configure access ports
        makeAccessPorts( nodeIndex, swIndex );

        // Configure trunk ports
        makeTrunkPorts( nodeIndex, swIndex );

        // Configure ingress remap table
        makeRemapTable( nodeIndex, swIndex );

        // Configure ingress RID and Rlan Route table
        makeRIDandRlanRouteTable( nodeIndex, swIndex );

        //
        // Switch GPIO:
        // according to http://lwbugs/3324387/10
        // Switch GPIO physical ID is the same as 0 based swIndex
        switches[nodeIndex][swIndex]->set_physicalid( swIndex );
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

bool vulcanSurrogate::getSwitchIndexByPhysicalId( uint32_t physicalId, uint32_t &index  )
{
    for (int i = 0; i < MAX_NUM_LWSWITCH_PER_NODE; i++ )
    {
        if ( mSwitchPorts[i][0].swPhysicalId == physicalId )
        {
            index = i;
            return true;
        }
    }
    return false;
}

void
vulcanSurrogate::updateSharedPartInfoLinkMasks( SharedPartInfoTable_t &partInfo )
{
    uint32_t s, g, swIndex;
    SharedPartSwitchInfoTable_t *switchInfo;
    SharedPartGpuInfoTable_t *gpuInfo;

    // Add switch trunk ports
    if ( partInfo.lwLinkIntraTrunkConnCount > 0 )
    {
        for ( s = 0; s < partInfo.switchCount; s++ )
        {
            switchInfo = &partInfo.switchInfo[s];
            if ( getSwitchIndexByPhysicalId( switchInfo->physicalId, swIndex ) == false )
            {
                printf("%s: failed to get switch index for physicalId 0x%x.\n",
                       __FUNCTION__, switchInfo->physicalId);
                continue;
            }

            for ( int p = 0; p < MAX_PORTS_PER_LWSWITCH; p++ )
            {
                if ( mSwitchPorts[swIndex][p].connectToSw == true )
                {
                    switchInfo->enabledLinkMask |= ( 1LL << p );
                }
            }
        }
    }

    // Add switch access ports and GPU ports
    for ( s = 0; s < partInfo.switchCount; s++ )
    {
        switchInfo = &partInfo.switchInfo[s];
        if ( getSwitchIndexByPhysicalId( switchInfo->physicalId, swIndex ) == false )
        {
            printf("%s: failed to get switch index for physicalId 0x%x.\n",
                   __FUNCTION__, switchInfo->physicalId);
            continue;
        }

        for ( g = 0; g < partInfo.gpuCount; g++ )
        {
            gpuInfo = &partInfo.gpuInfo[g];

            for ( int p = 0; p < MAX_PORTS_PER_LWSWITCH; p++ )
            {
                if ( ( mSwitchPorts[swIndex][p].connectToGpu == true ) &&
                     ( mSwitchPorts[swIndex][p].peerGpuPhysicalId == gpuInfo->physicalId ) )
                {
                    // add the switch access port
                    switchInfo->enabledLinkMask |= ( 1LL << p );

                    // add the gpu port
                    gpuInfo->enabledLinkMask |= ( 1LL << mSwitchPorts[swIndex][p].peerGpuPort );
                }
            }
        }
    }
}

void vulcanSurrogate::fillSharedPartInfoTable(int nodeIndex)
{
    return;
}

void vulcanSurrogate::fillSystemPartitionInfo(int nodeIndex)
{
    node *nodeInfo = nodes[nodeIndex];

    nodeSystemPartitionInfo *systemPartInfo = new nodeSystemPartitionInfo();

    // fill all the bare metal partition information

    // Vulcan Surrogate bare metal partition information
    bareMetalPartitionInfo *bareMetalPartInfo1 = systemPartInfo->add_baremetalinfo();
    partitionMetaDataInfo *bareMetaData1 = new partitionMetaDataInfo();
    bareMetaData1->set_gpucount( nodeInfo->gpu_size() );
    bareMetaData1->set_switchcount( nodeInfo->lwswitch_size() );
    // no intranode trunk connection
    bareMetaData1->set_lwlinkintratrunkconncount( 0 );
    // no internode trunk connection
    bareMetaData1->set_lwlinkintertrunkconncount( 0 );
    bareMetalPartInfo1->set_allocated_metadata( bareMetaData1 );

    systemPartInfo->set_name("VULCAN_SURROGATE");
    nodes[nodeIndex]->set_allocated_partitioninfo( systemPartInfo );

    systemPartInfo->set_version(FABRIC_PARTITION_VERSION);
}

void vulcanSurrogate::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
{
    int i, j;

    // Add GPUs
    for ( i = 0; i < gpuNum; i++ )
    {
        gpus[nodeIndex][i] = nodes[nodeIndex]->add_gpu();
        makeOneGpu( nodeIndex, i, ((nodeIndex * 8) + i), 0, 0x3F, 0xFFFFFFFF,
                    gpuFabricAddrBase[i], gpuFabricAddrRange[i],
                    gpuFlaAddrBase[i], gpuFlaAddrRange[i],
                    i, gpuTargetId[i]);

    }

    // Add LimeRocks
    for ( i = 0; i < lrNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }

    fillSystemPartitionInfo( nodeIndex );
}

void vulcanSurrogate::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    mNumSwitches = 4;
    mNumGpus = 8;

    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 8 GPUs, 4 LimeRock
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void vulcanSurrogate::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void vulcanSurrogate::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void vulcanSurrogate::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
