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
#include "deltaConfig.h"

// This is a full Delta Board topology with 16 GPUs and 8 LWSwitches

deltaConfig::deltaConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t targetId;

    mNumSwitches = 12;
    mNumGpus = 16;

    for ( targetId = 0; targetId < mNumGpus; targetId++ )
    {
        gpuTargetId[targetId] = targetId;
        gpuFabricAddrBase[targetId]  = (targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE) << 36;
        gpuFabricAddrRange[targetId] = FAB_ADDR_RANGE_16G * 8;

        gpuFlaAddrBase[targetId]  = (uint64_t)(targetId * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + LR_FIRST_FLA_RMAP_SLOT) << 36;
        gpuFlaAddrRange[targetId] = FAB_ADDR_RANGE_16G * 8;
    }

    mEnableWarmup  = false;
    mEnableSpray   = false;
    mMaxTargetId = 0;

    mSharedPartitionJsonFile = NULL;
    mSharedPartitiolwersion = 0;
    mNumSharedPartitions = DELTA_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS;

    time_t lwrtime = time(NULL);;
    struct tm *loc_time = localtime(&lwrtime);
    mSharedPartitionTimeStamp = asctime(loc_time);

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );
};

deltaConfig::~deltaConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};


void
deltaConfig::setConfig( bool enableWarmup, bool enableSpray, const char *sharedPartitionJsonFile )
{
    uint32_t swIndex, portIndex;
    const uint32_t numSwitchesPerBaseboard = 6;

    mEnableWarmup  = enableWarmup;
    mEnableSpray   = enableSpray;

    mNumSwitches = 12;
    mNumGpus = 16;

    if ( sharedPartitionJsonFile != NULL )
    {
        mSharedPartitionJsonFile = sharedPartitionJsonFile;
    }

    memset( mSwitchPorts, 0, sizeof(SwitchPort_t)*mNumSwitches*MAX_PORTS_PER_LWSWITCH );

    // Access ports
    // Switch 0, to all GPU port 2 and 3
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][25]  = {0,      0,            0,      0,        true,         0,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][24]  = {0,      0,            0,      1,        true,         0,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][27]  = {0,      0,            0,      0,        true,         1,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][26]  = {0,      0,            0,      1,        true,         1,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][34]  = {0,      0,            0,      0,        true,         2,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][35]  = {0,      0,            0,      1,        true,         2,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][32]  = {0,      0,            0,      0,        true,         3,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][33]  = {0,      0,            0,      1,        true,         3,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][10]  = {0,      0,            0,      0,        true,         4,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][11]  = {0,      0,            0,      1,        true,         4,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][29] =  {0,      0,            0,      0,        true,         5,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][28] =  {0,      0,            0,      1,        true,         5,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][30] =  {0,      0,            0,      0,        true,         6,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][31] =  {0,      0,            0,      1,        true,         6,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][8] =   {0,      0,            0,      0,        true,         7,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][9] =   {0,      0,            0,      1,        true,         7,                 3,           false,       0,            0,                0};

    // Switch 1, to all GPU port 8 and 9
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][12]  = {0,      0,            0,      0,        true,         0,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][13]  = {0,      0,            0,      1,        true,         0,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][34]  = {0,      0,            0,      0,        true,         1,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][35]  = {0,      0,            0,      1,        true,         1,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][8]   = {0,      0,            0,      0,        true,         2,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][9]   = {0,      0,            0,      1,        true,         2,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][15]  = {0,      0,            0,      0,        true,         3,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][14]  = {0,      0,            0,      1,        true,         3,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][11]  = {0,      0,            0,      0,        true,         4,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][10]  = {0,      0,            0,      1,        true,         4,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][24]  = {0,      0,            0,      0,        true,         5,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][25]  = {0,      0,            0,      1,        true,         5,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][27]  = {0,      0,            0,      0,        true,         6,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][26]  = {0,      0,            0,      1,        true,         6,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][32]  = {0,      0,            0,      0,        true,         7,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][33]  = {0,      0,            0,      1,        true,         7,                 9,           false,       0,            0,                0};

    // Switch 2, to all GPU port 4 and 5
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[2][30]  = {0,      0,            0,      0,        true,         0,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][31]  = {0,      0,            0,      1,        true,         0,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][13]  = {0,      0,            0,      0,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][12]  = {0,      0,            0,      1,        true,         1,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][34]  = {0,      0,            0,      0,        true,         2,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][35]  = {0,      0,            0,      1,        true,         2,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][14]  = {0,      0,            0,      0,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][15]  = {0,      0,            0,      1,        true,         3,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][28]  = {0,      0,            0,      0,        true,         4,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][29]  = {0,      0,            0,      1,        true,         4,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][8]   = {0,      0,            0,      0,        true,         5,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][9]   = {0,      0,            0,      1,        true,         5,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][26]  = {0,      0,            0,      0,        true,         6,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][27]  = {0,      0,            0,      1,        true,         6,                 5,           false,       0,            0,                0};
    mSwitchPorts[2][11]  = {0,      0,            0,      0,        true,         7,                 4,           false,       0,            0,                0};
    mSwitchPorts[2][10]  = {0,      0,            0,      1,        true,         7,                 5,           false,       0,            0,                0};

    // Switch 3, to all GPU port 0 and 1
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[3][9]   = {0,      0,            0,      0,        true,         0,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][8]   = {0,      0,            0,      1,        true,         0,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][30]  = {0,      0,            0,      0,        true,         1,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][31]  = {0,      0,            0,      1,        true,         1,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][29]  = {0,      0,            0,      0,        true,         2,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][28]  = {0,      0,            0,      1,        true,         2,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][34]  = {0,      0,            0,      0,        true,         3,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][35]  = {0,      0,            0,      1,        true,         3,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][27]  = {0,      0,            0,      0,        true,         4,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][26]  = {0,      0,            0,      1,        true,         4,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][10]  = {0,      0,            0,      0,        true,         5,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][11]  = {0,      0,            0,      1,        true,         5,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][25]  = {0,      0,            0,      0,        true,         6,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][24]  = {0,      0,            0,      1,        true,         6,                 1,           false,       0,            0,                0};
    mSwitchPorts[3][32]  = {0,      0,            0,      0,        true,         7,                 0,           false,       0,            0,                0};
    mSwitchPorts[3][33]  = {0,      0,            0,      1,        true,         7,                 1,           false,       0,            0,                0};

    // Switch 4, to all GPU port 10 and 11
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[4][31]  = {0,      0,            0,      0,        true,         0,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][30]  = {0,      0,            0,      1,        true,         0,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][15]  = {0,      0,            0,      0,        true,         1,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][14]  = {0,      0,            0,      1,        true,         1,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][12]  = {0,      0,            0,      0,        true,         2,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][13]  = {0,      0,            0,      1,        true,         2,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][34]  = {0,      0,            0,      0,        true,         3,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][35]  = {0,      0,            0,      1,        true,         3,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][27]  = {0,      0,            0,      0,        true,         4,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][26]  = {0,      0,            0,      1,        true,         4,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][29]  = {0,      0,            0,      0,        true,         5,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][28]  = {0,      0,            0,      1,        true,         5,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][10]  = {0,      0,            0,      0,        true,         6,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][11]  = {0,      0,            0,      1,        true,         6,                 11,          false,       0,            0,                0};
    mSwitchPorts[4][24]  = {0,      0,            0,      0,        true,         7,                 10,          false,       0,            0,                0};
    mSwitchPorts[4][25]  = {0,      0,            0,      1,        true,         7,                 11,          false,       0,            0,                0};

    // Switch 5, to all GPU port 6 and 7
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[5][12]  = {0,      0,            0,      0,        true,         0,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][13]  = {0,      0,            0,      1,        true,         0,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][25]  = {0,      0,            0,      0,        true,         1,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][24]  = {0,      0,            0,      1,        true,         1,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][26]  = {0,      0,            0,      0,        true,         2,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][27]  = {0,      0,            0,      1,        true,         2,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][15]  = {0,      0,            0,      0,        true,         3,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][14]  = {0,      0,            0,      1,        true,         3,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][29]  = {0,      0,            0,      0,        true,         4,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][28]  = {0,      0,            0,      1,        true,         4,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][10]  = {0,      0,            0,      0,        true,         5,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][11]  = {0,      0,            0,      1,        true,         5,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][31]  = {0,      0,            0,      0,        true,         6,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][30]  = {0,      0,            0,      1,        true,         6,                 7,           false,       0,            0,                0};
    mSwitchPorts[5][34]  = {0,      0,            0,      0,        true,         7,                 6,           false,       0,            0,                0};
    mSwitchPorts[5][35]  = {0,      0,            0,      1,        true,         7,                 7,           false,       0,            0,                0};

    // switch 6 to 11 are mirrors of switch 0 to 5
    // peerGpuPhysicalId is peerGpuPhysicalId + 8
    for ( swIndex = 6; swIndex < mNumSwitches; swIndex++ )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[swIndex-6][portIndex].connectToGpu == false )
                continue;

            mSwitchPorts[swIndex][portIndex].connectToGpu = true;
            mSwitchPorts[swIndex][portIndex].peerGpuPhysicalId = mSwitchPorts[swIndex-6][portIndex].peerGpuPhysicalId + 8;
            mSwitchPorts[swIndex][portIndex].peerGpuPort = mSwitchPorts[swIndex-6][portIndex].peerGpuPort;
            mSwitchPorts[swIndex][portIndex].rlanId = mSwitchPorts[swIndex-6][portIndex].rlanId;
        }
    }

    // Switches setting and trunk ports

    for ( swIndex = 0; swIndex < mNumSwitches; swIndex++ )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            mSwitchPorts[swIndex][portIndex].nodeId = 0;

            // Switch GPIO:
            // 8-13 on bottom baseboard
            // 24-29 on top baseboard
            if ( swIndex < numSwitchesPerBaseboard )
            {
                mSwitchPorts[swIndex][portIndex].swPhysicalId = 8 + swIndex;
            }
            else
            {
                mSwitchPorts[swIndex][portIndex].swPhysicalId = 18 + swIndex;
            }

            mSwitchPorts[swIndex][portIndex].swPort = portIndex;

            // Trunk port pairs
            // 0<->7,  1<->6, 2<->5, 3<->4, 16<->23, 17<->22, 18<->21, 19<->20
            if ( ( ( portIndex >= 0 )  && ( portIndex <= 7 ) ) ||
                 ( ( portIndex >= 16 ) && ( portIndex <= 23 ) ) )
            {
                mSwitchPorts[swIndex][portIndex].connectToGpu = false;
                mSwitchPorts[swIndex][portIndex].connectToSw  = true;
                mSwitchPorts[swIndex][portIndex].peerSwNodeId = 0;

                // trunk even port has rlanId 0, odd port has rlanId 1
                mSwitchPorts[swIndex][portIndex].rlanId = (portIndex % 2 == 0) ? 0 : 1;

                // peer switches
                // 8<->24, 9<->25, 10<->26, 11<->27, 12<->28, 13<->29
                if ( swIndex < numSwitchesPerBaseboard )
                {
                    mSwitchPorts[swIndex][portIndex].peerSwPhysicalId = mSwitchPorts[swIndex][portIndex].swPhysicalId + 16;
                }
                else
                {
                    mSwitchPorts[swIndex][portIndex].peerSwPhysicalId = mSwitchPorts[swIndex][portIndex].swPhysicalId - 16;
                }

                if ( portIndex <= 7 )
                {
                    mSwitchPorts[swIndex][portIndex].peerSwPort = 7 - portIndex;
                }
                else
                {
                    mSwitchPorts[swIndex][portIndex].peerSwPort = 39 - portIndex;
                }
            }
        } // portIndex
    } // swIndex

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

void deltaConfig::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t portIndex;
    int64_t mappedAddr, range, i, index;
    uint32_t gpuPhysicalId, targetId;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        // on all access ports on switch 0
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

void deltaConfig::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void deltaConfig::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

int
deltaConfig::getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex )
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
deltaConfig::getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPort )
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
deltaConfig::getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId)
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
deltaConfig::isGpuConnectedToSwitch( uint32_t swNodeId, uint32_t swIndex, uint32_t gpuNodeId, uint32_t gpuPhysicalId )
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

void deltaConfig::getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                       uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                       uint32_t *egressPortIndex, uint32_t *numEgressPorts, bool *isLastHop )
{
    int32_t i, portCount = 0;
    bool useTrunkPorts;

    *numEgressPorts = 0;
    *isLastHop = false;

    if ( isGpuConnectedToSwitch( nodeIndex, swIndex, destGpuNodeId, destGpuPhysicalId ) == true )
    {
        // destination GPU is on the same switch, no need to use trunk ports
        useTrunkPorts = false;
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
            // warmup is enabled, loopback from the same ingress access port
            if ( mEnableWarmup == true )
            {
                egressPortIndex[portCount] = ingressPortIndex;
                *numEgressPorts = 1;
                *isLastHop = true;
                return;
            }
            else
            {
                *numEgressPorts = 0;
                *isLastHop = true;
                return;
            }
        }

        // destGpu is connected to access ports that are different from the ingress port
        for ( i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( useTrunkPorts )
            {
                // useTrunkPorts is true, go to trunk ports
                if ( mSwitchPorts[swIndex][i].connectToSw == true )
                {
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
        return;
    }
}

void deltaConfig::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
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
                           // To select one egress trunk port from a set of 16 trunk ports
#if 0
                           // use destination GPU to select the trunk port with the same rlanId
                           // gpuPhysicalId * 2 + ingress port rlanId
                           // (gpuPhysicalId - 8) * 2 +  ingress port rlanId
                           selectedIndex = (gpuPhysicalId > 7) ?
                                   (gpuPhysicalId - 8) * 2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId :
                                   gpuPhysicalId * 2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId;
#endif
                           // use source GPU to select the trunk port with the same rlanId
                           // srcGpuPhysicalId * 2 + ingress port rlanId
                           // (srcGpuPhysicalId - 8) * 2 +  ingress port rlanId
                           uint32_t srcGpuPhysicalId = mSwitchPorts[swIndex][ingressPortIndex].peerGpuPhysicalId;
                           selectedIndex = (srcGpuPhysicalId > 7) ?
                                   (srcGpuPhysicalId - 8) * 2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId :
                                   srcGpuPhysicalId * 2 + mSwitchPorts[swIndex][ingressPortIndex].rlanId;

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

void deltaConfig::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void deltaConfig::makeAccessPorts( int nodeIndex, int swIndex )
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

void deltaConfig::makeTrunkPorts( int nodeIndex, int swIndex )
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

void deltaConfig::makeOneLwswitch( int nodeIndex, int swIndex )
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

        if ( swIndex < 6 )
        {
            switches[nodeIndex][swIndex]->set_physicalid( swIndex + 8 );
        } else {
            switches[nodeIndex][swIndex]->set_physicalid( swIndex - 6 + 24 );
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

bool deltaConfig::getSwitchIndexByPhysicalId( uint32_t physicalId, uint32_t &index  )
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
deltaConfig::updateSharedPartInfoLinkMasks( SharedPartInfoTable_t &partInfo )
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

void deltaConfig::fillSharedPartInfoTable(int nodeIndex)
{
    memset(mSharedVMPartInfo, 0, sizeof(SharedPartInfoTable_t)*DELTA_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS);
    uint32_t partitionId = 0;

    // fill in the LWSwitch and GPUs, the enabledLinkMask will be computed and filled in later

    mSharedVMPartInfo[partitionId] =
            // partitionId 0 - 16 GPUs, 12 LWSwitch, 96 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 16, 12, 96, 0,
              // all the GPUs, id, numlink, mask
              // all 12 GPU ports are used
              {   {0,   12,  0},
                  {1,   12,  0},
                  {2,   12,  0},
                  {3,   12,  0},
                  {4,   12,  0},
                  {5,   12,  0},
                  {6,   12,  0},
                  {7,   12,  0},
                  {8,   12,  0},
                  {9,   12,  0},
                  {10,  12,  0},
                  {11,  12,  0},
                  {12,  12,  0},
                  {13,  12,  0},
                  {14,  12,  0},
                  {15,  12,  0}  },

              // all the Switches, id, numlink, mask
              // 32 switch ports are used, 16 access + 16 trunk
              {   {0x08, 32, 0},
                  {0x09, 32, 0},
                  {0x0A, 32, 0},
                  {0x0B, 32, 0},
                  {0x0C, 32, 0},
                  {0x0D, 32, 0},
                  {0x18, 32, 0},
                  {0x19, 32, 0},
                  {0x1A, 32, 0},
                  {0x1B, 32, 0},
                  {0x1C, 32, 0},
                  {0x1D, 32, 0}  }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 1 - 8 GPUs, 6 LWSwitch base board 1 , 0 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 6, 0, 0,
              // all the GPUs, id, numlink, mask
              // all 12 GPU ports are used
              {   {0,    12, 0},
                  {1,    12, 0},
                  {2,    12, 0},
                  {3,    12, 0},
                  {4,    12, 0},
                  {5,    12, 0},
                  {6,    12, 0},
                  {7,    12, 0}  },

              // all the Switches, id, numlink, mask
              // 16 access ports are used
              {   {0x08, 16, 0},
                  {0x09, 16, 0},
                  {0x0A, 16, 0},
                  {0x0B, 16, 0},
                  {0x0C, 16, 0},
                  {0x0D, 16, 0}  }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 2 - 8 GPUs, 6 LWSwitch base board 2, 0 intra-trunk conn, 0 inter-trunk conn
            // all 12 GPU ports are used
            { partitionId, 8, 6, 0, 0,
              // all the GPUs, id, numlink, mask
              {   {8,    12,  0},
                  {9,    12,  0},
                  {10,   12,  0},
                  {11,   12,  0},
                  {12,   12,  0},
                  {13,   12,  0},
                  {14,   12,  0},
                  {15,   12,  0}, },

              // all the Switches, id, numlink, mask
              // 16 access ports are used
              {   {0x18, 16,  0},
                  {0x19, 16,  0},
                  {0x1A, 16,  0},
                  {0x1B, 16,  0},
                  {0x1C, 16,  0},
                  {0x1D, 16,  0} }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 3 - 8 GPUs
            // BB1 4 GPUs 0-3, BB2 4 GPUs 8-11
            // 6 LWSwitches BB1 & 6 LWSiwtches BB2, 96 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 12, 96, 0,
              // all the GPUs, id, numlink, mask
              // all 12 GPU ports are used
              {   // first 4 group from BB1 (refer to BB1 4 GPU partition below)
                  {0,    12, 0},
                  {1,    12, 0},
                  {2,    12, 0},
                  {3,    12, 0},
                  // first 4 group from BB2 (refer to BB2 4 GPU partition below)
                  {8,    12, 0},
                  {9,    12, 0},
                  {10,   12, 0},
                  {11,   12, 0} },

               // all the Switches, id, numlink, mask
               // 24 switch ports are used, 8 access + 16 trunk
               {// BB1 switches
                  {0x08, 24, 0},
                  {0x09, 24, 0},
                  {0x0A, 24, 0},
                  {0x0B, 24, 0},
                  {0x0C, 24, 0},
                  {0x0D, 24, 0},
                   // BB2 switches
                  {0x18, 24, 0},
                  {0x19, 24, 0},
                  {0x1A, 24, 0},
                  {0x1B, 24, 0},
                  {0x1C, 24, 0},
                  {0x1D, 24, 0} }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 4 - 8 GPUs
            // BB1 4 GPUs 4-7, BB2 4 GPUs 12-15
            // 6 LWSwitches BB1 & 6 LWSwitches BB2, 96 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 12, 96, 0,
                // all the GPUs, id, numlink, mask
                // all 12 GPU ports are used
                {
                     // first 4 group from BB1 (refer to BB1 4 GPU partition below)
                    {4,    12, 0},
                    {5,    12, 0},
                    {6,    12, 0},
                    {7,    12, 0},
                    // first 4 group from BB2 (refer to BB2 4 GPU partition below)
                    {12,   12, 0},
                    {13,   12, 0},
                    {14,   12, 0},
                    {15,   12, 0} },

                 // all the Switches, id, numlink, mask
                 // 24 switch ports are used, 8 access + 16 trunk
                 {
                     // BB1 switches.
                    {0x08, 24, 0},
                    {0x09, 24, 0},
                    {0x0A, 24, 0},
                    {0x0B, 24, 0},
                    {0x0C, 24, 0},
                    {0x0D, 24, 0},
                     // BB2 switches.
                    {0x18, 24, 0},
                    {0x19, 24, 0},
                    {0x1A, 24, 0},
                    {0x1B, 24, 0},
                    {0x1C, 24, 0},
                    {0x1D, 24, 0} }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 5 - 8 GPUs
            // BB1 4 GPUs 0-3, BB2 4 GPUs 12-15
            // 6 LWSwitches BB1 & 6 LWSwitches BB2, 96 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 12, 96, 0,
                // all the GPUs, id, numlink, mask
                {
                     // first 4 group from BB1 (refer to BB1 4 GPU partition below)
                    {0,    12, 0},
                    {1,    12, 0},
                    {2,    12, 0},
                    {3,    12, 0},
                    // first 4 group from BB2 (refer to BB2 4 GPU partition below)
                    {12,   12, 0},
                    {13,   12, 0},
                    {14,   12, 0},
                    {15,   12, 0} },

                 // all the Switches, id, numlink, mask
                 {
                     // BB1 switches. 24 = 8 access + 16 trunk
                    {0x08, 24, 0},
                    {0x09, 24, 0},
                    {0x0A, 24, 0},
                    {0x0B, 24, 0},
                    {0x0C, 24, 0},
                    {0x0D, 24, 0},
                     // BB2 switches. 24 = 8 access + 16 trunk
                    {0x18, 24, 0},
                    {0x19, 24, 0},
                    {0x1A, 24, 0},
                    {0x1B, 24, 0},
                    {0x1C, 24, 0},
                    {0x1D, 24, 0} }
            };

    partitionId++;
    mSharedVMPartInfo[partitionId] =
            // partitionId 6 - 8 GPUs
            // BB1 4 GPUs 4-7, BB2 4 GPUs 8-11
            // 6 LWSwitches BB1 & 6 LWSwitches BB2, 96 intra-trunk conn, 0 inter-trunk conn
            { partitionId, 8, 12, 96, 0,
                // all the GPUs, id, numlink, mask
                {
                     // first 4 group from BB1 (refer to BB1 4 GPU partition below)
                    {4,    12, 0},
                    {5,    12, 0},
                    {6,    12, 0},
                    {7,    12, 0},
                    // first 4 group from BB2 (refer to BB2 4 GPU partition below)
                    {8,    12, 0},
                    {9,    12, 0},
                    {10,   12, 0},
                    {11,   12, 0} },

                 // all the Switches, id, numlink, mask
                 {
                     // BB1 switches. 24 = 8 access + 16 trunk
                    {0x08, 24, 0},
                    {0x09, 24, 0},
                    {0x0A, 24, 0},
                    {0x0B, 24, 0},
                    {0x0C, 24, 0},
                    {0x0D, 24, 0},
                     // BB2 switches. 24 = 8 access + 16 trunk
                    {0x18, 24, 0},
                    {0x19, 24, 0},
                    {0x1A, 24, 0},
                    {0x1B, 24, 0},
                    {0x1C, 24, 0},
                    {0x1D, 24, 0} }
            };

    // 4 four GPU partitions, partitionId from 7 to 10
    // 4 GPUs,  6 LWSwitches, 0 intra-trunk conn, 0 inter-trunk conn
    // 12 ports on each GPU, 8 access ports on each switch
    partitionId++;
    for ( uint32_t physicalId = 0; physicalId < MAX_NUM_GPUS_PER_NODE;
          physicalId += 4, partitionId++)
    {
        mSharedVMPartInfo[partitionId].partitionId = partitionId;
        mSharedVMPartInfo[partitionId].gpuCount = 4;
        mSharedVMPartInfo[partitionId].switchCount = 6;
        mSharedVMPartInfo[partitionId].lwLinkIntraTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkInterTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].gpuInfo[0] = {physicalId,   12 , 0};
        mSharedVMPartInfo[partitionId].gpuInfo[1] = {physicalId+1,  12 , 0};
        mSharedVMPartInfo[partitionId].gpuInfo[2] = {physicalId+2,  12 , 0};
        mSharedVMPartInfo[partitionId].gpuInfo[3] = {physicalId+3,  12 , 0};

        if ( physicalId < 8 )
        {
            mSharedVMPartInfo[partitionId].switchInfo[0] = {0x08,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[1] = {0x09,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[2] = {0x0A,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[3] = {0x0B,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[4] = {0x0C,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[5] = {0x0D,  8, 0};
        }
        else
        {
            mSharedVMPartInfo[partitionId].switchInfo[0] = {0x18,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[1] = {0x19,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[2] = {0x1A,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[3] = {0x1B,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[4] = {0x1C,  8, 0};
            mSharedVMPartInfo[partitionId].switchInfo[5] = {0x1D,  8, 0};
        }
    }

    // 8 two GPU partitions, partitionId from 11 to 18
    // 2 GPUs,  6 LWSwitches, 0 intra-trunk conn, 0 inter-trunk conn
    // 12 ports on each GPU, 4 access ports on each switch
    for ( uint32_t physicalId = 0; physicalId < MAX_NUM_GPUS_PER_NODE;
          physicalId += 2, partitionId++)
    {
        mSharedVMPartInfo[partitionId].partitionId = partitionId;
        mSharedVMPartInfo[partitionId].gpuCount = 2;
        mSharedVMPartInfo[partitionId].switchCount = 6;
        mSharedVMPartInfo[partitionId].lwLinkIntraTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkInterTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].gpuInfo[0] = {physicalId,   12 , 0};
        mSharedVMPartInfo[partitionId].gpuInfo[1] = {physicalId+1, 12 , 0};

        if ( physicalId < 8 )
        {
            mSharedVMPartInfo[partitionId].switchInfo[0] = {0x08,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[1] = {0x09,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[2] = {0x0A,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[3] = {0x0B,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[4] = {0x0C,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[5] = {0x0D,  4, 0};
        }
        else
        {
            mSharedVMPartInfo[partitionId].switchInfo[0] = {0x18,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[1] = {0x19,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[2] = {0x1A,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[3] = {0x1B,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[4] = {0x1C,  4, 0};
            mSharedVMPartInfo[partitionId].switchInfo[5] = {0x1D,  4, 0};
        }
    }

    // 16 one GPU partitions, partitionId from 19 to 34
    // 1 GPUs, 0 LWSwitches, 0 intra-trunk conn, 0 inter-trunk conn
    for ( uint32_t physicalId = 0; physicalId < MAX_NUM_GPUS_PER_NODE;
          physicalId++, partitionId++)
    {
        mSharedVMPartInfo[partitionId].partitionId = partitionId;
        mSharedVMPartInfo[partitionId].gpuCount = 1;
        mSharedVMPartInfo[partitionId].switchCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkIntraTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].lwLinkInterTrunkConnCount = 0;
        mSharedVMPartInfo[partitionId].gpuInfo[0] = {physicalId, 0 , 0};
    }

    // update all switch and GPU link masks
    for ( partitionId = 0; partitionId < DELTA_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS; partitionId++ )
    {
        updateSharedPartInfoLinkMasks( mSharedVMPartInfo[partitionId] );
    }
}

void deltaConfig::fillSystemPartitionInfo(int nodeIndex)
{
    node *nodeInfo = nodes[nodeIndex];

    nodeSystemPartitionInfo *systemPartInfo = new nodeSystemPartitionInfo();

    // fill all the bare metal partition information

    // Delta bare metal partition information
    bareMetalPartitionInfo *bareMetalPartInfo1 = systemPartInfo->add_baremetalinfo();
    partitionMetaDataInfo *bareMetaData1 = new partitionMetaDataInfo();
    bareMetaData1->set_gpucount( nodeInfo->gpu_size() );
    bareMetaData1->set_switchcount( nodeInfo->lwswitch_size() );
    // total interanode trunk connections 96  (6 switch * 16)
    bareMetaData1->set_lwlinkintratrunkconncount( 96 );
    // no internode trunk connection for Delta
    bareMetaData1->set_lwlinkintertrunkconncount( 0 );
    bareMetalPartInfo1->set_allocated_metadata( bareMetaData1 );

    // Luna/HGX-next multi-host system bare metal partition information
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
    //    16        12       96
    //    8         6        0
    //    4         3        0
    //    2         1        0
    //    1         0        0
    //
    ptVMPartitionInfo *ptPartition1 = systemPartInfo->add_ptvirtualinfo();
    partitionMetaDataInfo *ptMetadata1 = new partitionMetaDataInfo();
    ptMetadata1->set_gpucount( 16 );
    ptMetadata1->set_switchcount( 12 );
    ptMetadata1->set_lwlinkintratrunkconncount( 96 );
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
    if ( mSharedPartitionJsonFile )
    {
        // the lwstmer provided partition definitions
        parsePartitionJsonFile( DELTA_NUM_TRUNK_LINKS );

        // update all switch and GPU link masks
        for ( uint i = 0; i < mNumSharedPartitions; i++ )
        {
            updateSharedPartInfoLinkMasks( mSharedVMPartInfo[i] );
        }
    }
    else
    {
        // Lwpu provided partition definitions
        fillSharedPartInfoTable( nodeIndex );
    }

    for ( uint32 idx = 0; idx < mNumSharedPartitions; idx++ )
    {
        SharedPartInfoTable_t partEntry = mSharedVMPartInfo[idx];
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

    if ( mSharedPartitionJsonFile )
    {
        mSystemPartInfo = systemPartInfo;
        systemPartInfo->set_name(mSharedPartitionJsonFile);
    }
    else
    {
        systemPartInfo->set_name("DGXA100_HGXA100");
    }
    systemPartInfo->set_version(FABRIC_PARTITION_VERSION);
    systemPartInfo->set_time(mSharedPartitionTimeStamp.c_str());
}

void deltaConfig::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
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

void deltaConfig::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;

    mNumSwitches = 12;
    mNumGpus = 16;

    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        // set up node IP address
        //nodeip << "192.168.0." << (nodeIndex + 1);
        //nodes[nodeIndex]->set_ipaddress( nodeip.str().c_str() );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 16 GPUs, 12 LimeRock
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void deltaConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void deltaConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void deltaConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
