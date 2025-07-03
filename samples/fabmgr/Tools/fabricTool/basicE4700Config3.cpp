#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "fabricConfig.h"
#include "basicE4700Config3.h"

basicE4700Config3::basicE4700Config3( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t targetId[4] = { 64, 576, 1088, 1600 };

    mNumSwitches = 2;
    mNumGpus = 4;

    for ( uint32_t i = 0; i < mNumGpus; i++ )
    {
        gpuTargetId[i] = targetId[i];
        gpuFabricAddrBase[i]  = gpuTargetId[i] << 36;
        gpuFabricAddrRange[i] = FAB_ADDR_RANGE_16G * 4;

        gpuFlaAddrBase[i]  = (uint64_t)(gpuTargetId[i] + LR_FIRST_FLA_RMAP_SLOT) << 36;
        gpuFlaAddrRange[i] = FAB_ADDR_RANGE_16G * 4;
    }

    mEnableWarmup  = false;
    mEnableSpray   = false;

    if ( gpuTargetId[3] < 512 )
    {
        mMaxTargetId = 0;
    }
    else if ( gpuTargetId[3] < 1024 )
    {
        mMaxTargetId = 1;
    }
    else
    {
        mMaxTargetId = 2;
    }

    memset(mSwitchPorts, 0, sizeof(SwitchPort_t) * MAX_PORTS_PER_LWSWITCH * 2);
};

basicE4700Config3::~basicE4700Config3()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void
basicE4700Config3::setE4700Config( bool enableWarmup, bool enableSpray )
{
    mEnableWarmup  = enableWarmup;
    mEnableSpray   = enableSpray;

    // Limerock 0
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][0]  = {0,      0,            0,      9,        true,         1,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][1]  = {0,      0,            1,      8,        true,         1,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][2]  = {0,      0,            2,      5,        true,         1,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][3]  = {0,      0,            3,      4,        true,         1,                 0,           false,       0,            0,                0};
    mSwitchPorts[0][4]  = {0,      0,            4,      11,       true,         1,                 11,          false,       0,            0,                0};
    mSwitchPorts[0][5]  = {0,      0,            5,      10,       true,         1,                 10,          false,       0,            0,                0};
    mSwitchPorts[0][6]  = {0,      0,            6,      7,        true,         1,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][7]  = {0,      0,            7,      6,        true,         1,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][8]  = {0,      0,            8,      4,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[0][9]  = {0,      0,            9,      5,        true,         1,                 5,           false,       0,            0,                0};
    mSwitchPorts[0][10] = {0,      0,            10,     6,        true,         1,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][11] = {0,      0,            11,     7,        true,         1,                 7,           false,       0,            0,                0};

    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][16] = {0,      0,            16,     10,       true,         0,                 10,          false,       0,            0,                0};
    mSwitchPorts[0][17] = {0,      0,            17,     11,       true,         0,                 11,          false,       0,            0,                0};
    mSwitchPorts[0][18] = {0,      0,            18,     6,        true,         0,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][19] = {0,      0,            19,     7,        true,         0,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][20] = {0,      0,            20,     8,        true,         0,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][21] = {0,      0,            21,     9,        true,         0,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][22] = {0,      0,            22,     4,        true,         0,                 4,           false,       0,            0,                0};
    mSwitchPorts[0][23] = {0,      0,            23,     5,        true,         0,                 5,           false,       0,            0,                0};
    mSwitchPorts[0][24] = {0,      0,            24,     3,        true,         0,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][25] = {0,      0,            25,     2,        true,         0,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][28] = {0,      0,            28,     1,        true,         0,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][29] = {0,      0,            29,     0,        true,         0,                 0,           false,       0,            0,                0};

    //                 nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][12] = {0,      0,            12,     0,        false,        0,                 0,           true,        0,            1,                12};
    mSwitchPorts[0][13] = {0,      0,            13,     1,        false,        0,                 0,           true,        0,            1,                13};
    mSwitchPorts[0][14] = {0,      0,            14,     2,        false,        0,                 0,           true,        0,            1,                14};
    mSwitchPorts[0][15] = {0,      0,            15,     3,        false,        0,                 0,           true,        0,            1,                15};
    mSwitchPorts[0][32] = {0,      0,            32,     4,        false,        0,                 0,           true,        0,            1,                32};
    mSwitchPorts[0][33] = {0,      0,            33,     5,        false,        0,                 0,           true,        0,            1,                33};

    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][26] = {0,      0,            26,     6,        false,        0,                 0,           true,        0,            1,                26};
    mSwitchPorts[0][27] = {0,      0,            27,     7,        false,        0,                 0,           true,        0,            1,                27};
    mSwitchPorts[0][30] = {0,      0,            30,     8,        false,        0,                 0,           true,        0,            1,                30};
    mSwitchPorts[0][31] = {0,      0,            31,     9,        false,        0,                 0,           true,        0,            1,                31};
    mSwitchPorts[0][34] = {0,      0,            34,     10,       false,        0,                 0,           true,        0,            1,                34};
    mSwitchPorts[0][35] = {0,      0,            35,     11,       false,        0,                 0,           true,        0,            1,                35};

    // Limerock 1
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][0]  = {0,      1,            0,      9,        true,         3,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][1]  = {0,      1,            1,      8,        true,         3,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][2]  = {0,      1,            2,      13,       true,         3,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][3]  = {0,      1,            3,      12,       true,         3,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][4]  = {0,      1,            4,      11,       true,         3,                 11,          false,       0,            0,                0};
    mSwitchPorts[1][5]  = {0,      1,            5,      10,       true,         3,                 10,          false,       0,            0,                0};
    mSwitchPorts[1][6]  = {0,      1,            6,      15,       true,         3,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][7]  = {0,      1,            7,      14,       true,         3,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][8]  = {0,      1,            8,      4,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][9]  = {0,      1,            9,      5,        true,         3,                 5,           false,       0,            0,                0};
    mSwitchPorts[1][10] = {0,      1,            10,     6,        true,         3,                 6,           false,       0,            0,                0};
    mSwitchPorts[1][11] = {0,      1,            11,     7,        true,         3,                 7,           false,       0,            0,                0};

    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][16] = {0,      1,            16,     10,       true,         2,                 10,          false,       0,            0,                0};
    mSwitchPorts[1][17] = {0,      1,            17,     11,       true,         2,                 11,          false,       0,            0,                0};
    mSwitchPorts[1][18] = {0,      1,            18,     6,        true,         2,                 6,           false,       0,            0,                0};
    mSwitchPorts[1][19] = {0,      1,            19,     7,        true,         2,                 7,           false,       0,            0,                0};
    mSwitchPorts[1][20] = {0,      1,            20,     8,        true,         2,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][21] = {0,      1,            21,     9,        true,         2,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][22] = {0,      1,            22,     4,        true,         2,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][23] = {0,      1,            23,     5,        true,         2,                 5,           false,       0,            0,                0};
    mSwitchPorts[1][24] = {0,      1,            24,     11,       true,         2,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][25] = {0,      1,            25,     10,       true,         2,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][28] = {0,      1,            28,     9,        true,         2,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][29] = {0,      1,            29,     8,        true,         2,                 0,           false,       0,            0,                0};

    //                 nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][12] = {0,      1,            12,     0,        false,        0,                 0,           true,        0,            0,                12};
    mSwitchPorts[1][13] = {0,      1,            13,     1,        false,        0,                 0,           true,        0,            0,                13};
    mSwitchPorts[1][14] = {0,      1,            14,     2,        false,        0,                 0,           true,        0,            0,                14};
    mSwitchPorts[1][15] = {0,      1,            15,     3,        false,        0,                 0,           true,        0,            0,                15};
    mSwitchPorts[1][32] = {0,      1,            32,     4,        false,        0,                 0,           true,        0,            0,                32};
    mSwitchPorts[1][33] = {0,      1,            33,     5,        false,        0,                 0,           true,        0,            0,                33};

    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][26] = {0,      1,            26,     6,        false,        0,                 0,           true,        0,            0,                26};
    mSwitchPorts[1][27] = {0,      1,            27,     7,        false,        0,                 0,           true,        0,            0,                27};
    mSwitchPorts[1][30] = {0,      1,            30,     8,        false,        0,                 0,           true,        0,            0,                30};
    mSwitchPorts[1][31] = {0,      1,            31,     9,        false,        0,                 0,           true,        0,            0,                31};
    mSwitchPorts[1][34] = {0,      1,            34,     10,       false,        0,                 0,           true,        0,            0,                34};
    mSwitchPorts[1][35] = {0,      1,            35,     11,       false,        0,                 0,           true,        0,            0,                35};


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

void basicE4700Config3::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t portIndex, index, i;
    int64_t mappedAddr, range;
    uint32_t gpuPhysicalId;

    if ( (nodeIndex == 0) && (swIndex < 2) )
    {
        // on all access ports on switch 0
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[swIndex][portIndex].connectToGpu == false )
                continue;

            for ( gpuPhysicalId = 0; gpuPhysicalId < mNumGpus; gpuPhysicalId++ )
            {
                // GPA map slot
                index = gpuFabricAddrBase[gpuPhysicalId] >> 36;
                range = gpuFabricAddrRange[gpuPhysicalId];

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
                                   index,               // target ID
                                   1,                   // rfunc = remap address. skip context and range checks
                                   0,                   // irl select
                                   0,                   // FAM swizzle
                                   0,                   // FAM mult2
                                   0);                  // FAM plane select

                // FLA map slot
                index = gpuFlaAddrBase[gpuPhysicalId] >> 36;
                range = gpuFlaAddrRange[gpuPhysicalId];

                makeOneRemapEntry( nodeIndex,           // nodeIndex
                                   swIndex,             // swIndex
                                   portIndex,           // portIndex
                                   index,               // rmap table index
                                   1,                   // entry valid
                                   gpuFlaAddrBase[gpuPhysicalId],   // 64 bits remap FLA to itsef, due to bug 2498189
                                   0,                   // context match
                                   0,                   // context mask
                                   0,                   // context replacement
                                   0,                   // address offset
                                   0,                   // address base
                                   0,                   // address limit
                                   (index-LR_FIRST_FLA_RMAP_SLOT), // target ID
                                   1,                   // remap FLA address. skip context and range checks
                                   0,                   // irl select
                                   0,                   // FAM swizzle
                                   0,                   // FAM mult2
                                   0);                  // FAM plane select
            }
        }
    }
}

void basicE4700Config3::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4700Config3::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4700Config3::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
{
    int portIndex;
    int egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS];
    int groupSelect[INGRESS_RLAN_MAX_GROUPS], groupSize[INGRESS_RLAN_MAX_GROUPS];
    int index = 64; // all GPUs has RID, RLAN index 64;

    if ( swIndex == 0 )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            memset(egressPort, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
            memset(vcMap, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
            memset(groupSelect, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);
            memset(groupSize, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);

            // switch 0, access port 3, 2, 7, 6 (GPU1)
            // To GPU0, 1, 2, 3

            if ( portIndex == 3 )
            {
                // To GPU 0,1,2,3 port 0
                egressPort[0] = 29;
                egressPort[1] = 3;
                egressPort[2] = 12;
                egressPort[3] = 12;
            }

            if ( portIndex == 2 )
            {
                // To GPU 0,1,2,3 port 1
                egressPort[0] = 28;
                egressPort[1] = 2;
                egressPort[2] = 13;
                egressPort[3] = 13;
            }

            if ( portIndex == 7 )
            {
                // To GPU 0,1,2,3 port 2
                egressPort[0] = 25;
                egressPort[1] = 7;
                egressPort[2] = 14;
                egressPort[3] = 14;
            }

            if ( portIndex == 6 )
            {
                // To GPU 0,1,2,3 port 3
                egressPort[0] = 24;
                egressPort[1] = 6;
                egressPort[2] = 15;
                egressPort[3] = 15;
            }


            // switch 0, access port 29, 28, 25, 24 (GPU0)
            // To GPU0, 1, 2, 3
            if ( portIndex == 29 )
            {
                // To GPU 0,1,2,3 port 0
                egressPort[0] = 29;
                egressPort[1] = 3;
                egressPort[2] = 12;
                egressPort[3] = 12;
            }

            if ( portIndex == 28 )
            {
                // To GPU 0,1,2,3 port 1
                egressPort[0] = 28;
                egressPort[1] = 2;
                egressPort[2] = 13;
                egressPort[3] = 13;
            }

            if ( portIndex == 25 )
            {
                // To GPU 0,1,2,3 port 2
                egressPort[0] = 25;
                egressPort[1] = 7;
                egressPort[2] = 14;
                egressPort[3] = 14;
            }

            if ( portIndex == 24 )
            {
                // To GPU 0,1,2,3 port 3
                egressPort[0] = 24;
                egressPort[1] = 6;
                egressPort[2] = 15;
                egressPort[3] = 15;
            }

            // switch 0, trunk port 12, 13, 14, 15
            // To GPU 0, 1
            if ( portIndex == 12 )
            {
                // To GPU 0,1 port 0
                egressPort[0] = 29;
                egressPort[1] = 3;
                egressPort[2] = 0;
                egressPort[3] = 0;
            }

            if ( portIndex == 13 )
            {
                // To GPU 0,1 port 0
                egressPort[0] = 28;
                egressPort[1] = 2;
                egressPort[2] = 0;
                egressPort[3] = 0;
            }

            if ( portIndex == 14 )
            {
                // To GPU 0,1 port 0
                egressPort[0] = 25;
                egressPort[1] = 7;
                egressPort[2] = 0;
                egressPort[3] = 0;
            }

            if ( portIndex == 15 )
            {
                // To GPU 0,1 port 0
                egressPort[0] = 24;
                egressPort[1] = 6;
                egressPort[2] = 0;
                egressPort[3] = 0;
            }

            // RID entry
            makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                  swIndex,          // swIndex
                                  portIndex,        // portIndex
                                  index,            // RID Route table index
                                  1,                // entry valid
                                  0x40,             // rmod (no special routing)
                                  4,                // number of ports
                                  vcMap,            // pointer to array of VC controls
                                  egressPort);      // pointer to array of ports

            for (int i = 0; i < INGRESS_RLAN_MAX_GROUPS; i++ )
            {
                groupSize[i] = 1;
                if ( (i >=0) && (i <= 3) )
                {
                    groupSelect[i] = 0;
                }

                if ( (i >=4) && (i <= 7) )
                {
                    groupSelect[i] = 1;
                }

                if ( (i >=8) && (i <= 11) )
                {
                    groupSelect[i] = 2;
                }

                if ( (i >=12) && (i <= 15) )
                {
                    groupSelect[i] = 3;
                }
            }

            // RLAN entry
            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   index,            // RLAN Route table index
                                   1,                // entry valid
                                   INGRESS_RLAN_MAX_GROUPS, // group count
                                   groupSelect,      // group select array
                                   groupSize );      // group size array
        }
    }

    if ( swIndex == 1 )
    {
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            memset(egressPort, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
            memset(vcMap, 0, sizeof(int)*INGRESS_RID_MAX_PORTS);
            memset(groupSelect, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);
            memset(groupSize, 0, sizeof(int)*INGRESS_RLAN_MAX_GROUPS);

            // switch 1, access port 3, 2, 7, 6 (GPU1)
            // To GPU0, 1, 2, 3

            if ( portIndex == 3 )
            {
                // To GPU 0,1,2,3 port 0
                egressPort[0] = 12;
                egressPort[1] = 12;
                egressPort[2] = 29;
                egressPort[3] = 3;
            }

            if ( portIndex == 2 )
            {
                // To GPU 0,1,2,3 port 1
                egressPort[0] = 13;
                egressPort[1] = 13;
                egressPort[2] = 28;
                egressPort[3] = 2;
            }

            if ( portIndex == 7 )
            {
                // To GPU 0,1,2,3 port 2
                egressPort[0] = 14;
                egressPort[1] = 14;
                egressPort[2] = 25;
                egressPort[3] = 7;
            }

            if ( portIndex == 6 )
            {
                // To GPU 0,1,2,3 port 3
                egressPort[0] = 15;
                egressPort[1] = 15;
                egressPort[2] = 24;
                egressPort[3] = 6;
            }


            // switch 1, access port 29, 28, 25, 24 (GPU0)
            // To GPU0, 1, 2, 3
            if ( portIndex == 29 )
            {
                // To GPU 0,1,2,3 port 0
                egressPort[0] = 12;
                egressPort[1] = 12;
                egressPort[2] = 29;
                egressPort[3] = 3;
            }

            if ( portIndex == 28 )
            {
                // To GPU 0,1,2,3 port 1
                egressPort[0] = 13;
                egressPort[1] = 13;
                egressPort[2] = 28;
                egressPort[3] = 2;
            }

            if ( portIndex == 25 )
            {
                // To GPU 0,1,2,3 port 2
                egressPort[0] = 14;
                egressPort[1] = 14;
                egressPort[2] = 25;
                egressPort[3] = 7;
            }

            if ( portIndex == 24 )
            {
                // To GPU 0,1,2,3 port 3
                egressPort[0] = 15;
                egressPort[1] = 15;
                egressPort[2] = 24;
                egressPort[3] = 6;
            }

            // switch 1, trunk port 12, 13, 14, 15
            // To GPU 2, 3
            if ( portIndex == 12 )
            {
                // To GPU 2,3 port 0
                egressPort[0] = 0;
                egressPort[1] = 0;
                egressPort[2] = 29;
                egressPort[3] = 3;
            }

            if ( portIndex == 13 )
            {
                // To GPU 2,3 port 1
                egressPort[0] = 0;
                egressPort[1] = 0;
                egressPort[2] = 28;
                egressPort[3] = 2;
            }

            if ( portIndex == 14 )
            {
                // To GPU 2,3 port 2
                egressPort[0] = 0;
                egressPort[1] = 0;
                egressPort[2] = 25;
                egressPort[3] = 7;
            }

            if ( portIndex == 15 )
            {
                // To GPU 2,3 port 3
                egressPort[0] = 0;
                egressPort[1] = 0;
                egressPort[2] = 24;
                egressPort[3] = 6;
            }

            // RID entry
            makeOneRIDRouteEntry( nodeIndex,        // nodeIndex
                                  swIndex,          // swIndex
                                  portIndex,        // portIndex
                                  index,            // RID Route table index
                                  1,                // entry valid
                                  0x40,             // rmod (no special routing)
                                  4,                // number of ports
                                  vcMap,            // pointer to array of VC controls
                                  egressPort);      // pointer to array of ports

            for (int i = 0; i < INGRESS_RLAN_MAX_GROUPS; i++ )
            {
                groupSize[i] = 1;
                if ( (i >=0) && (i <= 3) )
                {
                    groupSelect[i] = 0;
                }

                if ( (i >=4) && (i <= 7) )
                {
                    groupSelect[i] = 1;
                }

                if ( (i >=8) && (i <= 11) )
                {
                    groupSelect[i] = 2;
                }

                if ( (i >=12) && (i <= 15) )
                {
                    groupSelect[i] = 3;
                }
            }

            // RLAN entry
            makeOneRLANRouteEntry( nodeIndex,        // nodeIndex
                                   swIndex,          // swIndex
                                   portIndex,        // portIndex
                                   index,            // RLAN Route table index
                                   1,                // entry valid
                                   INGRESS_RLAN_MAX_GROUPS, // group count
                                   groupSelect,      // group select array
                                   groupSize );      // group size array
        }
    }
}

void basicE4700Config3::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4700Config3::makeAccessPorts( int nodeIndex, int swIndex )
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

void basicE4700Config3::makeTrunkPorts( int nodeIndex, int swIndex )
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

void basicE4700Config3::makeOneLwswitch( int nodeIndex, int swIndex )
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
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4700Config3::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
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
}

void basicE4700Config3::makeNodes()
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
            // node 0 has 4 GPUs, 2 LimeRock
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void basicE4700Config3::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4700Config3::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4700Config3::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
