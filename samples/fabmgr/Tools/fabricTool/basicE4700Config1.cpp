#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "fabricConfig.h"
#include "basicE4700Config1.h"

//
// This is a single E4700 board topology with 0, 1 or 2 GPUs and 1 LWSwitches
// https://confluence.lwpu.com/display/LIMEROCK/LR10+Bringup#LR10Bringup-Loopback/LoopoutModulesConnectivityDiagram
//

basicE4700Config1::basicE4700Config1( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    // physicalId/targetId 0
    gpuTargetId[0] = 0;
    gpuFabricAddrBase[0]  = (uint64_t)(gpuTargetId[0] * NUM_INGR_RMAP_ENTRIES_PER_AMPERE) << 36;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 8;

    gpuFlaAddrBase[0]  = (uint64_t)(gpuTargetId[0] * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + LR_FIRST_FLA_RMAP_SLOT) << 36;
    gpuFlaAddrRange[0] = FAB_ADDR_RANGE_16G * 8;

    // physicalId/targetId 1
    gpuTargetId[1] = 1;
    gpuFabricAddrBase[1]  = (uint64_t)(gpuTargetId[1] * NUM_INGR_RMAP_ENTRIES_PER_AMPERE) << 36;
    gpuFabricAddrRange[1] = FAB_ADDR_RANGE_16G * 8;

    gpuFlaAddrBase[1]  = (uint64_t)(gpuTargetId[1] * NUM_INGR_RMAP_ENTRIES_PER_AMPERE + LR_FIRST_FLA_RMAP_SLOT) << 36;
    gpuFlaAddrRange[1] = FAB_ADDR_RANGE_16G * 8;

    mSxm4Slot0   = NotPopulated;
    mSxm4Slot1   = NotPopulated;
    mExaMAXslot0 = NotPopulated;
    mExaMAXslot1 = NotPopulated;

    mEnableWarmup  = false;
    mEnableSpray   = false;
    mUseTrunkPorts = false;

    if ( gpuTargetId[1] < 512 )
    {
        mMaxTargetId = 0;
    }
    else if ( gpuTargetId[1] < 1024 )
    {
        mMaxTargetId = 1;
    }
    else
    {
        mMaxTargetId = 2;
    }

    memset(mSwitchPorts, 0, sizeof(SwitchPort_t) * MAX_PORTS_PER_LWSWITCH);
};

basicE4700Config1::~basicE4700Config1()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void
basicE4700Config1::setE4700Config(E4700EndpointType_t sxm4Slot0, E4700EndpointType_t sxm4Slot1,
                                  E4700EndpointType_t exaMAXslot0, E4700EndpointType_t exaMAXslot1,
                                  bool enableWarmup, bool enableSpray, bool useTrunkPort)
{
    mEnableWarmup  = enableWarmup;
    mEnableSpray   = enableSpray;
    mUseTrunkPorts = useTrunkPort;

    mSxm4Slot0   = sxm4Slot0;
    mSxm4Slot1   = sxm4Slot1;
    mExaMAXslot0 = exaMAXslot0;
    mExaMAXslot1 = exaMAXslot1;

    // construct the switch port connections
    if ( mSxm4Slot0 == PG506 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[0]  = {0,      0,            0,      9,        true,         0,                 9,           false,       0,            0,                0};
        mSwitchPorts[1]  = {0,      0,            1,      8,        true,         0,                 8,           false,       0,            0,                0};
        mSwitchPorts[2]  = {0,      0,            2,      1,        true,         0,                 1,           false,       0,            0,                0};
        mSwitchPorts[3]  = {0,      0,            3,      0,        true,         0,                 0,           false,       0,            0,                0};
        mSwitchPorts[4]  = {0,      0,            4,      11,       true,         0,                 11,          false,       0,            0,                0};
        mSwitchPorts[5]  = {0,      0,            5,      10,       true,         0,                 10,          false,       0,            0,                0};
        mSwitchPorts[6]  = {0,      0,            6,      3,        true,         0,                 3,           false,       0,            0,                0};
        mSwitchPorts[7]  = {0,      0,            7,      2,        true,         0,                 2,           false,       0,            0,                0};
        mSwitchPorts[8]  = {0,      0,            8,      4,        true,         0,                 4,           false,       0,            0,                0};
        mSwitchPorts[9]  = {0,      0,            9,      5,        true,         0,                 5,           false,       0,            0,                0};
        mSwitchPorts[10] = {0,      0,            10,     6,        true,         0,                 6,           false,       0,            0,                0};
        mSwitchPorts[11] = {0,      0,            11,     7,        true,         0,                 7,           false,       0,            0,                0};
    }
    else if ( mSxm4Slot0 == E4702 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[0]  = {0,      0,            0,      0,        false,        0,                 0,           true,        0,            0,                4};
        mSwitchPorts[1]  = {0,      0,            1,      1,        false,        0,                 0,           true,        0,            0,                5};
        mSwitchPorts[2]  = {0,      0,            2,      2,        false,        0,                 0,           true,        0,            0,                9};
        mSwitchPorts[3]  = {0,      0,            3,      3,        false,        0,                 0,           true,        0,            0,                8};
        mSwitchPorts[4]  = {0,      0,            4,      4,        false,        0,                 0,           true,        0,            0,                0};
        mSwitchPorts[5]  = {0,      0,            5,      5,        false,        0,                 0,           true,        0,            0,                1};
        mSwitchPorts[6]  = {0,      0,            6,      6,        false,        0,                 0,           true,        0,            0,                11};
        mSwitchPorts[7]  = {0,      0,            7,      7,        false,        0,                 0,           true,        0,            0,                10};
        mSwitchPorts[8]  = {0,      0,            8,      8,        false,        0,                 0,           true,        0,            0,                3};
        mSwitchPorts[9]  = {0,      0,            9,      9,        false,        0,                 0,           true,        0,            0,                2};
        mSwitchPorts[10] = {0,      0,            10,     10,       false,        0,                 0,           true,        0,            0,                7};
        mSwitchPorts[11] = {0,      0,            11,     11,       false,        0,                 0,           true,        0,            0,                6};
    }

    if ( mSxm4Slot1 == PG506 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[16] = {0,      0,            16,     10,       true,         1,                 10,          false,       0,            0,                0};
        mSwitchPorts[17] = {0,      0,            17,     11,       true,         1,                 11,          false,       0,            0,                0};
        mSwitchPorts[18] = {0,      0,            18,     6,        true,         1,                 6,           false,       0,            0,                0};
        mSwitchPorts[19] = {0,      0,            19,     7,        true,         1,                 7,           false,       0,            0,                0};
        mSwitchPorts[20] = {0,      0,            20,     8,        true,         1,                 8,           false,       0,            0,                0};
        mSwitchPorts[21] = {0,      0,            21,     9,        true,         1,                 9,           false,       0,            0,                0};
        mSwitchPorts[22] = {0,      0,            22,     4,        true,         1,                 4,           false,       0,            0,                0};
        mSwitchPorts[23] = {0,      0,            23,     5,        true,         1,                 5,           false,       0,            0,                0};
        mSwitchPorts[24] = {0,      0,            24,     3,        true,         1,                 3,           false,       0,            0,                0};
        mSwitchPorts[25] = {0,      0,            25,     2,        true,         1,                 2,           false,       0,            0,                0};
        mSwitchPorts[28] = {0,      0,            28,     1,        true,         1,                 1,           false,       0,            0,                0};
        mSwitchPorts[29] = {0,      0,            29,     0,        true,         1,                 0,           false,       0,            0,                0};
    }
    else if ( mSxm4Slot1 == E4702 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[16] = {0,      0,            16,     0,        false,        0,                 0,           true,        0,            0,                20};
        mSwitchPorts[17] = {0,      0,            17,     1,        false,        0,                 0,           true,        0,            0,                21};
        mSwitchPorts[18] = {0,      0,            18,     2,        false,        0,                 0,           true,        0,            0,                25};
        mSwitchPorts[19] = {0,      0,            19,     3,        false,        0,                 0,           true,        0,            0,                24};
        mSwitchPorts[20] = {0,      0,            20,     4,        false,        0,                 0,           true,        0,            0,                16};
        mSwitchPorts[21] = {0,      0,            21,     5,        false,        0,                 0,           true,        0,            0,                17};
        mSwitchPorts[22] = {0,      0,            22,     6,        false,        0,                 0,           true,        0,            0,                29};
        mSwitchPorts[23] = {0,      0,            23,     7,        false,        0,                 0,           true,        0,            0,                28};
        mSwitchPorts[24] = {0,      0,            24,     8,        false,        0,                 0,           true,        0,            0,                19};
        mSwitchPorts[25] = {0,      0,            25,     9,        false,        0,                 0,           true,        0,            0,                18};
        mSwitchPorts[28] = {0,      0,            28,     10,       false,        0,                 0,           true,        0,            0,                23};
        mSwitchPorts[29] = {0,      0,            29,     11,       false,        0,                 0,           true,        0,            0,                22};
    }

    if ( mExaMAXslot0 == E4705 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[12] = {0,      0,            12,     0,        false,        0,                 0,           true,        0,            0,                12};
        mSwitchPorts[13] = {0,      0,            13,     1,        false,        0,                 0,           true,        0,            0,                13};
        mSwitchPorts[14] = {0,      0,            14,     2,        false,        0,                 0,           true,        0,            0,                14};
        mSwitchPorts[15] = {0,      0,            15,     3,        false,        0,                 0,           true,        0,            0,                15};
        mSwitchPorts[32] = {0,      0,            32,     4,        false,        0,                 0,           true,        0,            0,                32};
        mSwitchPorts[33] = {0,      0,            33,     5,        false,        0,                 0,           true,        0,            0,                33};
    }

    if ( mExaMAXslot1 == E4705 )
    {
        //                  nodeId, swPhysicalId, swPort, rlandId, connectToGpu, peerGpuPhysicalId, peerGpuPort, connectToSw, peerSwNodeId, peerSwPhysicalId, peerSwPort
        mSwitchPorts[26] = {0,      0,            26,     6,        false,        0,                 0,           true,        0,            0,                26};
        mSwitchPorts[27] = {0,      0,            27,     7,        false,        0,                 0,           true,        0,            0,                27};
        mSwitchPorts[30] = {0,      0,            30,     8,        false,        0,                 0,           true,        0,            0,                30};
        mSwitchPorts[31] = {0,      0,            31,     9,        false,        0,                 0,           true,        0,            0,                31};
        mSwitchPorts[34] = {0,      0,            34,     10,       false,        0,                 0,           true,        0,            0,                34};
        mSwitchPorts[35] = {0,      0,            35,     11,       false,        0,                 0,           true,        0,            0,                35};
    }

    printf("index nodeId swPhysicalId swPort connectToGpu peerGpuPhysicalId peerGpuPort connectToSw peerSwNodeId peerSwPhysicalId peerSwPort\n");
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
    {
        printf("%5d %6d %12d %6d %12d %17d %11d %11d %12d %16d %10d\n",
                i, mSwitchPorts[i].nodeId, mSwitchPorts[i].swPhysicalId, mSwitchPorts[i].swPort, mSwitchPorts[i].connectToGpu,
                mSwitchPorts[i].peerGpuPhysicalId, mSwitchPorts[i].peerGpuPort, mSwitchPorts[i].connectToSw, mSwitchPorts[i].peerSwNodeId,
                mSwitchPorts[i].peerSwPhysicalId, mSwitchPorts[i].peerSwPort);
    }
}

void basicE4700Config1::makeRemapTable( int nodeIndex, int swIndex )
{
    int32_t portIndex;
    int64_t mappedAddr, range, i, index;
    uint32_t gpuPhysicalId, targetId;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        int trunkPortCount = getNumTrunkPorts(nodeIndex, swIndex);


        // on all access ports
        for ( portIndex = 0; portIndex < MAX_PORTS_PER_LWSWITCH; portIndex++ )
        {
            if ( mSwitchPorts[portIndex].connectToGpu == false )
                continue;

            for ( gpuPhysicalId = 0; gpuPhysicalId < 2; gpuPhysicalId++ )
            {
                if ( ( ( gpuPhysicalId == 0 ) && ( mSxm4Slot0 == PG506 ) ) ||
                     ( ( gpuPhysicalId == 1 ) && ( mSxm4Slot1 == PG506 ) ) )
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
}

void basicE4700Config1::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4700Config1::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}

int
basicE4700Config1::getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex )
{
    int count = 0;
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[i].connectToSw == true ) )
        {
            count++;
        }
    }
    return count;
}

int
basicE4700Config1::getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPort )
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[i].swPort == swPort ) )
        {
            return i;
        }
    }

    return -1;
}

int
basicE4700Config1::getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId)
{
    for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i ++ )
    {
        if ( ( mSwitchPorts[i].nodeId == nodeIndex ) &&
             ( mSwitchPorts[i].swPhysicalId == swIndex ) &&
             ( mSwitchPorts[i].connectToGpu == true ) &&
             ( mSwitchPorts[i].peerGpuPort == rlanId ) )
        {
            return i;
        }
    }

    return -1;
}

void basicE4700Config1::getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                             uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                             uint32_t *egressPortIndex, uint32_t *numEgressPorts)
{
    int32_t i, portCount = 0;
    bool useTrunkPorts = mUseTrunkPorts;

    *numEgressPorts = 0;

    if ( getNumTrunkPorts(nodeIndex, swIndex) == 0 )
    {
        // there is no trunk port
        useTrunkPorts = false;
    }

    int ingressPortIndex = getSwitchPortIndex(nodeIndex, swIndex, ingressPortNum);
    if ( ( ingressPortIndex < 0 ) || ( ingressPortIndex > MAX_PORTS_PER_LWSWITCH ) )
    {
        // cannot find the ingress port
        printf("%s: Cannot find the ingress port swNodeId %d, swPhysicalId %d, ingressPortNum %d.\n",
                __FUNCTION__, nodeIndex, swIndex, ingressPortNum);
        return;
    }

    SwitchPort_t ingressPort = mSwitchPorts[ingressPortIndex];

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
                return;
            }

            // warmup is disabled, go out trunk ports
            // if both E4705 and E4702 are plugged in, use E4705 ports first
            for ( i = MAX_PORTS_PER_LWSWITCH - 1; i >=0; i-- )
            {
                if ( mSwitchPorts[i].connectToSw == false )
                {
                    // not a trunk port
                    continue;
                }

                egressPortIndex[portCount] = i;
                portCount++;
            }

            *numEgressPorts = portCount > MAX_LWLINKS_PER_GPU ? MAX_LWLINKS_PER_GPU : portCount;
            return;
        }

        // destGpu is connected to access ports that are different from the ingress port
        for ( i = MAX_PORTS_PER_LWSWITCH - 1; i >= 0; i-- )
        {
            if ( useTrunkPorts )
            {
                // useTrunkPorts is true, go to trunk ports
                if ( mSwitchPorts[i].connectToSw == true )
                {
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
            else
            {
                // useTrunkPorts is false, go to access ports connected to destGpu
                if ( ( mSwitchPorts[i].connectToGpu ) &&
                     ( mSwitchPorts[i].nodeId == destGpuNodeId ) &&
                     ( mSwitchPorts[i].peerGpuPhysicalId == destGpuPhysicalId ) )
                {
                    egressPortIndex[portCount] = i;
                    portCount++;
                }
            }
        }
        *numEgressPorts = portCount > MAX_LWLINKS_PER_GPU ? MAX_LWLINKS_PER_GPU : portCount;
        return;
    }

    // ingress port is a trunk port
    if ( ingressPort.connectToSw )
    {
        // go to access ports connected to destGpu
        for ( i = MAX_PORTS_PER_LWSWITCH - 1; i >= 0; i-- )
        {
            if ( ( mSwitchPorts[i].connectToGpu ) &&
                 ( mSwitchPorts[i].nodeId == destGpuNodeId ) &&
                 ( mSwitchPorts[i].peerGpuPhysicalId == destGpuPhysicalId ) )
            {
                egressPortIndex[portCount] = i;
                portCount++;
            }

        }
        *numEgressPorts = portCount > MAX_LWLINKS_PER_GPU ? MAX_LWLINKS_PER_GPU : portCount;
        return;
    }
}

void basicE4700Config1::makeRIDandRlanRouteTable( int nodeIndex, int swIndex )
{
    uint32_t i, egressPortCount;
    uint32_t egressPortIndex[MAX_LWLINKS_PER_GPU];
    int egressPort[INGRESS_RID_MAX_PORTS], vcMap[INGRESS_RID_MAX_PORTS];

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int gpuPhysicalId = 0; gpuPhysicalId < 2; gpuPhysicalId++ )
        {
            if ( ( ( gpuPhysicalId == 0 ) && ( mSxm4Slot0 == PG506 ) ) ||
                 ( ( gpuPhysicalId == 1 ) && ( mSxm4Slot1 == PG506 ) ) )
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

                    getEgressPortsToGpu( nodeIndex, swIndex, portIndex,
                                         nodeIndex, gpuPhysicalId,
                                         egressPortIndex, &egressPortCount);

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
                        // select only one port with the same rlanId from the egress port list
                        uint32_t j;
                        for ( j = 0; j < egressPortCount; j++ )
                        {
                            if ( mSwitchPorts[egressPortIndex[j]].rlanId == mSwitchPorts[ingressPortIndex].rlanId )
                            {
                                i = j;
                                break;
                            }
                        }

                        if ( j >= egressPortCount )
                        {
                            printf("Failed to find an egress port with rlandId %d\n",
                                    mSwitchPorts[ingressPortIndex].rlanId);
                            break;
                        }

                        vcMap[0] = 0;
                        egressPort[0] = egressPortIndex[i];
                        printf("select egress port i %d, number %d\n",
                               i, egressPort[0]);

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

                    }

                } // for portIndex
            }
        } // for gpuPhysicalId
    }
}

void basicE4700Config1::makeGangedLinkTable( int nodeIndex, int swIndex )
{
    return;
}

void basicE4700Config1::makeAccessPorts( int nodeIndex, int swIndex )
{
    uint32_t farPeerID, farPeerTargetID;

    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[i].connectToGpu == false )
                continue;

            makeOneAccessPort( nodeIndex,
                               swIndex,
                               mSwitchPorts[i].swPort,            // portIndex
                               0,                                 // farNodeID
                               mSwitchPorts[i].peerGpuPhysicalId, // farPeerID
                               mSwitchPorts[i].peerGpuPort,       // farPortNum
                               DC_COUPLED,                        // phyMode
                               mSwitchPorts[i].peerGpuPhysicalId, // farPeerTargetID
                               mSwitchPorts[i].rlanId,            // rlanID, set to connected GPU port number
                               mMaxTargetId);                     // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4700Config1::makeTrunkPorts( int nodeIndex, int swIndex )
{
    if ( (nodeIndex == 0) && (swIndex == 0) )
    {
        for ( int i = 0; i < MAX_PORTS_PER_LWSWITCH; i++ )
        {
            if ( mSwitchPorts[i].connectToSw == false )
                continue;

            makeOneTrunkPort( nodeIndex,
                              swIndex,
                              mSwitchPorts[i].swPort,           // portIndex
                              0,                                // farNodeID
                              mSwitchPorts[i].peerSwPhysicalId, // farSwitchID
                              mSwitchPorts[i].peerSwPort,       // farPortNum
                              DC_COUPLED,                       // phyMode
                              mMaxTargetId);                    // maxTargetID
        }
    }
    else
    {
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4700Config1::makeOneLwswitch( int nodeIndex, int swIndex )
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
        printf("%s: Invalid LimeRock nodeIndex %d swIndex %d.\n", __FUNCTION__, nodeIndex, swIndex);
    }
}

void basicE4700Config1::makeOneNode( int nodeIndex, int gpuNum, int lrNum )
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

void basicE4700Config1::makeNodes()
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
            // node 0 has 2 GPUs, 1 LimeRock
            makeOneNode( nodeIndex, 2, 1);
            break;

        default:
            printf("%s: Invalid nodeIndex %d.\n", __FUNCTION__, nodeIndex);
            break;
        }
    }
}

// implement willow virtual functions
void basicE4700Config1::makeOneWillow( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4700Config1::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    return;
}

void basicE4700Config1::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    return;
}
