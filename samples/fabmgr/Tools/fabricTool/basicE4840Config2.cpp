#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "FMDeviceProperty.h"
#include "fabricConfig.h"
#include "basicE4840Config2.h"

//
// This is a single E4840 board topology with 0, 1 or 2 GPUs and 1 LWSwitches
// https://confluence.lwpu.com/display/PLTFMSOLGRP/E4840+Board+Architecture
//

basicE4840Config2::basicE4840Config2( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    uint64_t targetId;

    mNumSwitches = 2;
    mNumGpus = 4;

    for (uint32_t targetId = 0; targetId < mNumGpus; targetId++) {

        gpuTargetId[targetId] = targetId;

        gpuFabricAddrBase[targetId]  = FMDeviceProperty::getGpaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFabricAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuGpaEgmAddrBase[targetId]  = FMDeviceProperty::getGpaEgmFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuGpaEgmAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);

        gpuFlaAddrBase[targetId]  = FMDeviceProperty::getFlaFromTargetId(LWSWITCH_ARCH_TYPE_LS10, targetId);
        gpuFlaAddrRange[targetId] = FMDeviceProperty::getAddressRangePerGpu(LWSWITCH_ARCH_TYPE_LS10);
    }

    mEnableWarmup  = false;
    mEnableSpray   = false;
    mUseTrunkPorts = false;

    memset(mSwitchPorts, 0, sizeof(SwitchPort_t) * MAX_PORTS_PER_LWSWITCH);
};

basicE4840Config2::~basicE4840Config2()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void
basicE4840Config2::setE4840Config(bool enableWarmup, bool enableSpray, bool useTrunkPort)
{
    mEnableWarmup  = enableWarmup;
    mEnableSpray   = enableSpray;
    mUseTrunkPorts = useTrunkPort;

    // GPU 0 on switch 0
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][0]  = {0,      0,            0,      0,        true,         0,                 0,           false,       0,            0,                0};
    mSwitchPorts[0][1]  = {0,      0,            1,      1,        true,         0,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][2]  = {0,      0,            2,      2,        true,         0,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][3]  = {0,      0,            3,      3,        true,         0,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][4]  = {0,      0,            4,      4,        true,         0,                 4,           false,       0,            0,                0};
    mSwitchPorts[0][5]  = {0,      0,            5,      5,        true,         0,                 5,           false,       0,            0,                0};
    mSwitchPorts[0][6]  = {0,      0,            6,      6,        true,         0,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][7]  = {0,      0,            7,      7,        true,         0,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][8]  = {0,      0,            8,      8,        true,         0,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][9]  = {0,      0,            9,      9,        true,         0,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][10] = {0,      0,            10,    10,        true,         0,                10,           false,       0,            0,                0};
    mSwitchPorts[0][11] = {0,      0,            11,    11,        true,         0,                11,           false,       0,            0,                0};
    mSwitchPorts[0][12] = {0,      0,            12,    12,        true,         0,                12,           false,       0,            0,                0};
    mSwitchPorts[0][13] = {0,      0,            13,    13,        true,         0,                13,           false,       0,            0,                0};
    mSwitchPorts[0][14] = {0,      0,            14,    14,        true,         0,                14,           false,       0,            0,                0};
    mSwitchPorts[0][15] = {0,      0,            15,    15,        true,         0,                15,           false,       0,            0,                0};
    mSwitchPorts[0][16] = {0,      0,            16,    16,        true,         0,                16,           false,       0,            0,                0};
    mSwitchPorts[0][17] = {0,      0,            17,    17,        true,         0,                17,           false,       0,            0,                0};

    // GPU 1 on switch 0
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][18] = {0,      0,            18,    17,        true,         1,                17,           false,       0,            0,                0};
    mSwitchPorts[0][19] = {0,      0,            19,    16,        true,         1,                16,           false,       0,            0,                0};
    mSwitchPorts[0][20] = {0,      0,            20,    15,        true,         1,                15,           false,       0,            0,                0};
    mSwitchPorts[0][21] = {0,      0,            21,    14,        true,         1,                14,           false,       0,            0,                0};
    mSwitchPorts[0][22] = {0,      0,            22,    13,        true,         1,                13,           false,       0,            0,                0};
    mSwitchPorts[0][23] = {0,      0,            23,    12,        true,         1,                12,           false,       0,            0,                0};
    mSwitchPorts[0][24] = {0,      0,            24,    11,        true,         1,                11,           false,       0,            0,                0};
    mSwitchPorts[0][25] = {0,      0,            25,    10,        true,         1,                10,           false,       0,            0,                0};
    mSwitchPorts[0][26] = {0,      0,            26,     9,        true,         1,                 9,           false,       0,            0,                0};
    mSwitchPorts[0][27] = {0,      0,            27,     8,        true,         1,                 8,           false,       0,            0,                0};
    mSwitchPorts[0][28] = {0,      0,            28,     7,        true,         1,                 7,           false,       0,            0,                0};
    mSwitchPorts[0][29] = {0,      0,            29,     6,        true,         1,                 6,           false,       0,            0,                0};
    mSwitchPorts[0][30] = {0,      0,            30,     5,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[0][31] = {0,      0,            31,     4,        true,         1,                 4,           false,       0,            0,                0};
    mSwitchPorts[0][32] = {0,      0,            32,     3,        true,         1,                 3,           false,       0,            0,                0};
    mSwitchPorts[0][33] = {0,      0,            33,     2,        true,         1,                 2,           false,       0,            0,                0};
    mSwitchPorts[0][34] = {0,      0,            34,     1,        true,         1,                 1,           false,       0,            0,                0};
    mSwitchPorts[0][35] = {0,      0,            35,     0,        true,         1,                 0,           false,       0,            0,                0};

    // GPU 2 on switch 1
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][0]  = {0,      0,            0,      0,        true,         2,                 0,           false,       0,            0,                0};
    mSwitchPorts[1][1]  = {0,      0,            1,      1,        true,         2,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][2]  = {0,      0,            2,      2,        true,         2,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][3]  = {0,      0,            3,      3,        true,         2,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][4]  = {0,      0,            4,      4,        true,         2,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][5]  = {0,      0,            5,      5,        true,         2,                 5,           false,       0,            0,                0};
    mSwitchPorts[1][6]  = {0,      0,            6,      6,        true,         2,                 6,           false,       0,            0,                0};
    mSwitchPorts[1][7]  = {0,      0,            7,      7,        true,         2,                 7,           false,       0,            0,                0};
    mSwitchPorts[1][8]  = {0,      0,            8,      8,        true,         2,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][9]  = {0,      0,            9,      9,        true,         2,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][10] = {0,      0,            10,    10,        true,         2,                10,           false,       0,            0,                0};
    mSwitchPorts[1][11] = {0,      0,            11,    11,        true,         2,                11,           false,       0,            0,                0};
    mSwitchPorts[1][12] = {0,      0,            12,    12,        true,         2,                12,           false,       0,            0,                0};
    mSwitchPorts[1][13] = {0,      0,            13,    13,        true,         2,                13,           false,       0,            0,                0};
    mSwitchPorts[1][14] = {0,      0,            14,    14,        true,         2,                14,           false,       0,            0,                0};
    mSwitchPorts[1][15] = {0,      0,            15,    15,        true,         2,                15,           false,       0,            0,                0};
    mSwitchPorts[1][16] = {0,      0,            16,    16,        true,         2,                16,           false,       0,            0,                0};
    mSwitchPorts[1][17] = {0,      0,            17,    17,        true,         2,                17,           false,       0,            0,                0};

    // GPU 3 on switch 2
    //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][18] = {0,      0,            18,    17,        true,         3,                17,           false,       0,            0,                0};
    mSwitchPorts[1][19] = {0,      0,            19,    16,        true,         3,                16,           false,       0,            0,                0};
    mSwitchPorts[1][20] = {0,      0,            20,    15,        true,         3,                15,           false,       0,            0,                0};
    mSwitchPorts[1][21] = {0,      0,            21,    14,        true,         3,                14,           false,       0,            0,                0};
    mSwitchPorts[1][22] = {0,      0,            22,    13,        true,         3,                13,           false,       0,            0,                0};
    mSwitchPorts[1][23] = {0,      0,            23,    12,        true,         3,                12,           false,       0,            0,                0};
    mSwitchPorts[1][24] = {0,      0,            24,    11,        true,         3,                11,           false,       0,            0,                0};
    mSwitchPorts[1][25] = {0,      0,            25,    10,        true,         3,                10,           false,       0,            0,                0};
    mSwitchPorts[1][26] = {0,      0,            26,     9,        true,         3,                 9,           false,       0,            0,                0};
    mSwitchPorts[1][27] = {0,      0,            27,     8,        true,         3,                 8,           false,       0,            0,                0};
    mSwitchPorts[1][28] = {0,      0,            28,     7,        true,         3,                 7,           false,       0,            0,                0};
    mSwitchPorts[1][29] = {0,      0,            29,     6,        true,         3,                 6,           false,       0,            0,                0};
    mSwitchPorts[1][30] = {0,      0,            30,     5,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][31] = {0,      0,            31,     4,        true,         3,                 4,           false,       0,            0,                0};
    mSwitchPorts[1][32] = {0,      0,            32,     3,        true,         3,                 3,           false,       0,            0,                0};
    mSwitchPorts[1][33] = {0,      0,            33,     2,        true,         3,                 2,           false,       0,            0,                0};
    mSwitchPorts[1][34] = {0,      0,            34,     1,        true,         3,                 1,           false,       0,            0,                0};
    mSwitchPorts[1][35] = {0,      0,            35,     0,        true,         3,                 0,           false,       0,            0,                0};

    // OSFP connections, only use 18 trunks to make routing symmetrical
    // OSFP0 switch0 connects to OSPF0 switch1
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][36] = {0,      0,            36,     17,      false,        0,                 0,           true,        0,            1,                36};
    mSwitchPorts[0][37] = {0,      0,            37,     16,      false,        0,                 0,           true,        0,            1,                37};
    mSwitchPorts[0][38] = {0,      0,            38,     15,      false,        0,                 0,           true,        0,            1,                38};
    mSwitchPorts[0][39] = {0,      0,            39,     14,      false,        0,                 0,           true,        0,            1,                39};


    // OSFP1 switch0 connects to OSPF1 switch1
    mSwitchPorts[0][40] = {0,      0,            40,     13,      false,        0,                 0,           true,        0,            1,                40};
    mSwitchPorts[0][41] = {0,      0,            41,     12,      false,        0,                 0,           true,        0,            1,                41};
    mSwitchPorts[0][42] = {0,      0,            42,     11,      false,        0,                 0,           true,        0,            1,                42};
    mSwitchPorts[0][43] = {0,      0,            43,     10,      false,        0,                 0,           true,        0,            1,                43};

    // OSFP2 switch0 connects to OSPF2 switch1
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][44] = {0,      0,            44,     9,        false,        0,                 0,          true,        0,            1,                44};
    mSwitchPorts[0][45] = {0,      0,            45,     8,        false,        0,                 0,          true,        0,            1,                45};
    mSwitchPorts[0][46] = {0,      0,            46,     7,        false,        0,                 0,          true,        0,            1,                46};
    mSwitchPorts[0][47] = {0,      0,            47,     6,        false,        0,                 0,          true,        0,            1,                47};

    // OSFP3 switch0 connects to OSPF3 switch1
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][48] = {0,      0,            48,     5,        false,        0,                 0,          true,        0,            1,                48};
    mSwitchPorts[0][48] = {0,      0,            49,     4,        false,        0,                 0,          true,        0,            1,                49};
    mSwitchPorts[0][50] = {0,      0,            50,     3,        false,        0,                 0,          true,        0,            1,                50};
    mSwitchPorts[0][51] = {0,      0,            51,     2,        false,        0,                 0,          true,        0,            1,                51};

    // OSFP6 switch0 connects to OSPF6 switch1
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[0][62] = {0,      0,            62,     1,        false,        0,                 0,           true,       0,            1,                62};
    mSwitchPorts[0][63] = {0,      0,            63,     0,        false,        0,                 0,           true,       0,            1,                63};

    // OSFP0 switch1 connects to OSPF0 switch0
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][36] = {0,      0,            36,     17,      false,        0,                 0,           true,        0,            0,                36};
    mSwitchPorts[1][37] = {0,      0,            37,     16,      false,        0,                 0,           true,        0,            0,                37};
    mSwitchPorts[1][38] = {0,      0,            38,     15,      false,        0,                 0,           true,        0,            0,                38};
    mSwitchPorts[1][39] = {0,      0,            39,     14,      false,        0,                 0,           true,        0,            0,                39};


    // OSFP1 switch1 connects to OSPF0 switch0
    mSwitchPorts[1][40] = {0,      0,            40,     13,      false,        0,                 0,           true,        0,            0,                40};
    mSwitchPorts[1][41] = {0,      0,            41,     12,      false,        0,                 0,           true,        0,            0,                41};
    mSwitchPorts[1][42] = {0,      0,            42,     11,      false,        0,                 0,           true,        0,            0,                42};
    mSwitchPorts[1][43] = {0,      0,            43,     10,      false,        0,                 0,           true,        0,            0,                43};

    // OSFP2 switch1 connects to OSPF2 switch0
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][44] = {0,      0,            44,     9,        false,        0,                 0,          true,        0,            0,                44};
    mSwitchPorts[1][45] = {0,      0,            45,     8,        false,        0,                 0,          true,        0,            0,                45};
    mSwitchPorts[1][46] = {0,      0,            46,     7,        false,        0,                 0,          true,        0,            0,                46};
    mSwitchPorts[1][47] = {0,      0,            47,     6,        false,        0,                 0,          true,        0,            0,                47};

    // OSFP3 switch1 connects to OSPF3 switch0
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][48] = {0,      0,            48,     5,        false,        0,                 0,          true,        0,            0,                48};
    mSwitchPorts[1][48] = {0,      0,            49,     4,        false,        0,                 0,          true,        0,            0,                49};
    mSwitchPorts[1][50] = {0,      0,            50,     3,        false,        0,                 0,          true,        0,            0,                50};
    mSwitchPorts[1][51] = {0,      0,            51,     2,        false,        0,                 0,          true,        0,            0,                51};

    // OSFP6 switch1 connects to OSPF6 switch0
    //                nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
    mSwitchPorts[1][62] = {0,      0,            62,     1,        false,        0,                 0,          true,        0,            0,                62};
    mSwitchPorts[1][63] = {0,      0,            63,     0,        false,        0,                 0,          true,        0,            0,                63};

    printf("index nodeId swPhysicalId swPort connectToGpu peerGpuPhysicalId peerGpuPort connectToSw peerSwNodeId peerSwPhysicalId peerSwPort\n");
    for ( int j = 0; j < 1; j++ )
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

void basicE4840Config2::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr, range;
    GPU *gpu;
    uint32_t targetId;

    if ( (nodeIndex == 0) && (swIndex < (int)mNumSwitches) )
    {
        // on all access ports
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[swIndex][portIndex].connectToGpu == false )
                continue;

            for ( targetId = 0; targetId < mNumGpus; targetId++ )
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
}

void basicE4840Config2::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4840Config2::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

int
basicE4840Config2::getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex )
{
    int count = 0;
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToSw == true ) )
        {
            count++;
        }
    }
    return count;
}

int
basicE4840Config2::getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPort )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[swIndex][i].swPort == swPort ) )
        {
            return i;
        }
    }

    return -1;
}

int
basicE4840Config2::getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId)
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
    {
        if ( ( mSwitchPorts[swIndex][i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[swIndex][i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[swIndex][i].connectToGpu == true ) &&
             ( mSwitchPorts[swIndex][i].peerGpuPort == rlanId ) )
        {
            return i;
        }
    }

    return -1;
}

bool
basicE4840Config2::isGpuConnectedToSwitch( uint32_t swNodeId, uint32_t swIndex, uint32_t gpuNodeId, uint32_t gpuPhysicalId )
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

void basicE4840Config2::getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                       uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                       uint32_t *egressPortIndex, uint32_t *numEgressPorts, bool *isLastHop )
{
    uint32_t portCount = 0;
    bool useTrunkPorts;

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

            // warmup is disabled
            *numEgressPorts = 0;
            *isLastHop = true;
            return;
        }

        // destGpu is connected to access ports that are different from the ingress port
        // OSFP ports are from 26 to 63, counting down favors OSPF ports than E4824 trunk ports
        for ( int i = MAX_PORTS_PER_LWSWITCH - 1; i >= 0; i-- )
        {
            if ( useTrunkPorts )
            {
                // useTrunkPorts is true, go to trunk ports
                if ( mSwitchPorts[swIndex][i].connectToSw == true )
                {
                    egressPortIndex[portCount] = i;
                    portCount++;
                    *isLastHop = false;
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
                    *isLastHop = true;
                }
            }
        }
        *numEgressPorts = portCount > FMDeviceProperty::getLWLinksPerGpu(LWSWITCH_ARCH_TYPE_LS10) ?
                FMDeviceProperty::getLWLinksPerGpu(LWSWITCH_ARCH_TYPE_LS10) : portCount;
        return;
    }

    // ingress port is a trunk port
    if ( ingressPort.connectToSw )
    {
        // go to access ports connected to destGpu
        for ( int i = MAX_PORTS_PER_LWSWITCH - 1; i >= 0; i-- )
        {
            if ( ( mSwitchPorts[swIndex][i].connectToGpu ) &&
                 ( mSwitchPorts[swIndex][i].nodeId == destGpuNodeId ) &&
                 ( mSwitchPorts[swIndex][i].peerGpuPhysicalId == destGpuPhysicalId ) )
            {
                egressPortIndex[portCount] = i;
                portCount++;
            }

        }
        *numEgressPorts = portCount > FMDeviceProperty::getLWLinksPerGpu(LWSWITCH_ARCH_TYPE_LS10) ?
                FMDeviceProperty::getLWLinksPerGpu(LWSWITCH_ARCH_TYPE_LS10) : portCount;
        *isLastHop = true;
        return;
    }
}

void basicE4840Config2::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
{
    uint32_t i, egressPortCount;
    uint32_t egressPortIndex[MAX_PORTS_PER_LWSWITCH];
    int egressPort[MAX_PORTS_PER_LWSWITCH], vcMap[MAX_PORTS_PER_LWSWITCH];
    int groupSelect[INGRESS_RLAN_MAX_GROUPS], groupSize[INGRESS_RLAN_MAX_GROUPS];
    bool isLastHop;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int gpuPhysicalId = 0; gpuPhysicalId < (int)mNumGpus; gpuPhysicalId++ )
        {

            for ( int portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
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

                memset(egressPortIndex, 0, sizeof(int)*MAX_PORTS_PER_LWSWITCH);
                memset(egressPort, 0, sizeof(int)*MAX_PORTS_PER_LWSWITCH);
                memset(vcMap, 0, sizeof(int)*MAX_PORTS_PER_LWSWITCH);
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
                    printf("\t");
                }

                if ( mEnableSpray == false )
                {
                    // no spray
                    uint32_t j, selectedIndex = 0xFFFF;

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

                    if ( selectedIndex > FMDeviceProperty::getLWLinksPerGpu(LWSWITCH_ARCH_TYPE_LS10) )
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
                    int rmod = 0x40;

                    if ( egressPortCount > INGRESS_RLAN_MAX_GROUPS )
                    {
                        // there could be 18 or more egress ports
                        // but INGRESS_RLAN_MAX_GROUPS is only 16
                        egressPortCount = INGRESS_RLAN_MAX_GROUPS;
                    }

                    for ( uint32_t j = 0; j < egressPortCount; j++ )
                    {
                        groupSelect[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = j;
                        groupSize[mSwitchPorts[swIndex][egressPortIndex[j]].rlanId] = 1;
                        egressPort[j] = mSwitchPorts[swIndex][egressPortIndex[j]].swPort;
                    }

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
                } // for portIndex
            }
        } // for gpuPhysicalId
    }
}

void basicE4840Config2::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4840Config2::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToGpu == false )
                continue;

            makeOneAccessPort( nodeIndex,
                               swIndex,
                               mSwitchPorts[swIndex][i].swPort,            // portIndex
                               0,                                          // farNodeID
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerID
                               mSwitchPorts[swIndex][i].peerGpuPort,       // farPortNum
                               DC_COUPLED,                                 // phyMode
                               mSwitchPorts[swIndex][i].peerGpuPhysicalId, // farPeerTargetID
                               mSwitchPorts[swIndex][i].rlanId,            // rlanID, set to connected GPU port number
                               0);                                         // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid Switch nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4840Config2::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[swIndex][i].connectToSw == false )
                continue;

            makeOneTrunkPort( nodeIndex,
                              swIndex,
                              mSwitchPorts[swIndex][i].swPort,           // portIndex
                              0,                                         // farNodeID
                              mSwitchPorts[swIndex][i].peerSwPhysicalId, // farSwitchID
                              mSwitchPorts[swIndex][i].peerSwPort,       // farPortNum
                              DC_COUPLED,                                // phyMode
                              0);                                        // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid Switch nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4840Config2::makeOneLwswitch( int nodeIndex, int swIndex )
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

        // Configure ingress RID and Rlan Route table
        makeRIDandRlanRouteTable( nodeIndex, swIndex );
    }
    else
    {
        printf("%s: Invalid Switch nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4840Config2::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
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

    // Add switch
    for ( i = 0; i < lrNum; i++ )
    {
        switches[nodeIndex][i] = nodes[nodeIndex]->add_lwswitch();
        switches[nodeIndex][i]->set_version( FABRIC_MANAGER_VERSION );
        makeOneLwswitch( nodeIndex, i);
    }
}

void basicE4840Config2::makeNodes()
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
            // node 0 has 4 GPUs, 2 Switch
            makeOneNode( nodeIndex, mNumGpus, mNumSwitches);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void basicE4840Config2::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4840Config2::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4840Config2::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
