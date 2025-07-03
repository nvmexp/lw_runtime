#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "basicE3600Config8.h"

basicE3600Config8::basicE3600Config8( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    gpuFabricAddrBase[0]  = (uint64_t)0x6AC << 34;
    gpuFabricAddrRange[0] = FAB_ADDR_RANGE_16G * 4;

    gpuFabricAddrBase[1]  = (uint64_t)0x0AB << 34;
    gpuFabricAddrRange[1] = FAB_ADDR_RANGE_16G * 4;

    gpuFabricAddrBase[2]  = (uint64_t)0x8 << 34;
    gpuFabricAddrRange[2] = FAB_ADDR_RANGE_16G * 4;

    gpuFabricAddrBase[3]  = (uint64_t)0x1F00 << 34;
    gpuFabricAddrRange[3] = FAB_ADDR_RANGE_16G * 4;
};

basicE3600Config8::~basicE3600Config8()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void basicE3600Config8::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr = 0, range = 0;
    GPU *gpu;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            // To gpu[1]
            gpu   = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[1]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[2]
            gpu   = gpus[nodeIndex][2];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[3]
            gpu   = gpus[nodeIndex][3];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            // To gpu[0]
            gpu   = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[0]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[2]
            gpu   = gpus[nodeIndex][2];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[3]
            gpu   = gpus[nodeIndex][3];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Trunk port 6, 7
        for ( portIndex = 6; portIndex <= 7; portIndex++ )
        {
            // To gpu[0]
            gpu = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[0]
                    mappedAddr = 0;
                }
                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[1]
            gpu = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[1]
                    mappedAddr = 0;
                }
                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Trunk port 14, 15
        for ( portIndex = 14; portIndex <= 15; portIndex++ )
        {
            // To gpu[0]
            gpu = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[0]
                    mappedAddr = 0;
                }
                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[1]
            gpu = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[1]
                    mappedAddr = 0;
                }
                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    } //if ( (nodeIndex == 0) && (willowIndex == 0) )

    if ( (nodeIndex == 0) && (willowIndex == 1) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            // To gpu[3]
            gpu   = gpus[nodeIndex][3];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[3]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[0]
            gpu   = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[1]
            gpu   = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            // To gpu[2]
            gpu   = gpus[nodeIndex][2];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[2]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[0]
            gpu   = gpus[nodeIndex][0];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[1]
            gpu   = gpus[nodeIndex][1];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x11000000,    // vcModeValid7_0, port 6 and 7
                                        0x11000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Trunk port 6, 7
        for ( portIndex = 6; portIndex <= 7; portIndex++ )
        {
            // To gpu[2]
            gpu   = gpus[nodeIndex][2];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[2]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[3]
            gpu   = gpus[nodeIndex][3];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[3]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }

        // Trunk port 14, 15
        for ( portIndex = 14; portIndex <= 15; portIndex++ )
        {
            // To gpu[2]
            gpu   = gpus[nodeIndex][2];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[2]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00111111,    // vcModeValid7_0, port 0 to 5
                                        0x00000000,    // vcModeValid15_8
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }

            // To gpu[3]
            gpu   = gpus[nodeIndex][3];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    //mappedAddr = ( (int64_t)index ) << 34;
                    // last hop to gpu[3]
                    mappedAddr = 0;
                }

                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0
                                        0x00111111,    // vcModeValid15_8, port 8 to 13
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    } //if ( (nodeIndex == 0) && (willowIndex == 1) )
}

void basicE3600Config8::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, i, index, enpointID, outPortNum;
    accessPort *outPort;

    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            // gpuIndex 1 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 1
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                 nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 2 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 2
            enpointID = 2;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }

            // gpuIndex 3 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 3
            enpointID = 3;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 0
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 2 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 2
            enpointID = 2;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }

            // gpuIndex 3 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 3
            enpointID = 3;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }
        }

        // Trunk port 6, 7
        for ( portIndex = 6; portIndex <= 7; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 0
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 1 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 1
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }
        }

        // Trunk port 14, 15
        for ( portIndex = 14; portIndex <= 15; portIndex++ )
        {
            // gpuIndex 0 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 0
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 1 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 1
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }
        }
    } // if ( (nodeIndex == 0) && (willowIndex == 0) )

    if ( (nodeIndex == 0) && (willowIndex == 1) )
    {
        // Access port 0 to 5
        for ( portIndex = 0; portIndex <= 5; portIndex++ )
        {
            // gpuIndex 3 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 3
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 0 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 0
            enpointID = 0;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }

            // gpuIndex 1 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 1
            enpointID = 1;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }
        }

        // Access port 8 to 13
        for ( portIndex = 8; portIndex <= 13; portIndex++ )
        {
            // gpuIndex 2 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 2
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 0 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 0
            enpointID = 0;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }

            // gpuIndex 1 is connected to outPort trunk 6 and 7, 14 and 15
            // index is the requesterLinkId of gpuIndex 1
            enpointID = 1;
            for ( index = (enpointID*6); index <= ((enpointID*6)+5); index++ )
            {
                 makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                          willowIndex,   // willowIndex
                                          portIndex,     // portIndex
                                          index,         // Ingress resq table index
                                          0,             // routePolicy
                                          0x11000000,    // vcModeValid7_0, port 6 and 7
                                          0x11000000,    // vcModeValid15_8, port 14 and 15
                                          0x00000000,    // vcModeValid17_16
                                          1);            // entryValid
            }
        }

        // Trunk port 6, 7
        for ( portIndex = 6; portIndex <= 7; portIndex++ )
        {
            // gpuIndex 2 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 2
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 3 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 3
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }
        }

        // Trunk port 14, 15
        for ( portIndex = 14; portIndex <= 15; portIndex++ )
        {
            // gpuIndex 2 is connected to outPort access port 0 to 5
            // index is the requesterLinkId of gpuIndex 2
            for ( outPortNum = 0; outPortNum <= 5; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             1 << (4*outPortNum), // vcModeValid7_0
                                             0x00000000,    // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }

            // gpuIndex 3 is connected to outPort access port 8 to 13
            // index is the requesterLinkId of gpuIndex 3
            for ( outPortNum = 8; outPortNum <= 13; outPortNum++ )
            {
                outPort = accesses[nodeIndex][willowIndex][outPortNum];
                if ( outPort && outPort->has_farpeerid() && outPort->has_farportnum() )
                {
                    // index is the requesterLinkId
                    index = outPort->farpeerid() * 6 + outPort->farportnum();

                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0
                                             1 << (4*(outPortNum - 8)), // vcModeValid15_8
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else
                {
                    PRINT_ERROR("%d,%d,%d", "invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                                nodeIndex, willowIndex, outPortNum);
                }
            }
        }
    } //if ( (nodeIndex == 0) && (willowIndex == 1) )
}

void basicE3600Config8::makeGangedLinkTable( int nodeIndex, int willowIndex )
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

void basicE3600Config8::makeAccessPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        0,          0,        0,        0,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          1,        0,        0,        0,         DC_COUPLED);
        makeOneAccessPort(0,        0,          2,        0,        0,        5,         DC_COUPLED);
        makeOneAccessPort(0,        0,          3,        0,        0,        4,         DC_COUPLED);
        makeOneAccessPort(0,        0,          4,        0,        0,        2,         DC_COUPLED);
        makeOneAccessPort(0,        0,          5,        0,        0,        3,         DC_COUPLED);
        makeOneAccessPort(0,        0,          8,        0,        1,        2,         DC_COUPLED);
        makeOneAccessPort(0,        0,          9,        0,        1,        3,         DC_COUPLED);
        makeOneAccessPort(0,        0,          10,       0,        1,        4,         DC_COUPLED);
        makeOneAccessPort(0,        0,          11,       0,        1,        5,         DC_COUPLED);
        makeOneAccessPort(0,        0,          12,       0,        1,        1,         DC_COUPLED);
        makeOneAccessPort(0,        0,          13,       0,        1,        0,         DC_COUPLED);
    }
    else if ( (nodeIndex == 0) && (willowIndex == 1) )
    {
        //                nodeIndex willowIndex portIndex farNodeID farPeerID farPortNum portMode
        makeOneAccessPort(0,        1,          0,        0,        2,        1,         DC_COUPLED);
        makeOneAccessPort(0,        1,          1,        0,        2,        0,         DC_COUPLED);
        makeOneAccessPort(0,        1,          2,        0,        2,        5,         DC_COUPLED);
        makeOneAccessPort(0,        1,          3,        0,        2,        4,         DC_COUPLED);
        makeOneAccessPort(0,        1,          4,        0,        2,        2,         DC_COUPLED);
        makeOneAccessPort(0,        1,          5,        0,        2,        3,         DC_COUPLED);
        makeOneAccessPort(0,        1,          8,        0,        3,        2,         DC_COUPLED);
        makeOneAccessPort(0,        1,          9,        0,        3,        3,         DC_COUPLED);
        makeOneAccessPort(0,        1,          10,       0,        3,        4,         DC_COUPLED);
        makeOneAccessPort(0,        1,          11,       0,        3,        5,         DC_COUPLED);
        makeOneAccessPort(0,        1,          12,       0,        3,        1,         DC_COUPLED);
        makeOneAccessPort(0,        1,          13,       0,        3,        0,         DC_COUPLED);
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config8::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( (nodeIndex == 0) && (willowIndex == 0) )
    {
        //               nodeIndex willowIndex portIndex farNodeID farSwitchID farPortNum portMode
        makeOneTrunkPort(nodeIndex,willowIndex,6,        0,        1,          6,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,7,        0,        1,          7,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,14,       0,        1,         14,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,15,       0,        1,         15,         AC_COUPLED);

    }
    else if ( (nodeIndex == 0) && (willowIndex == 1) )
    {
        //               nodeIndex willowIndex portIndex farNodeID farSwitchID farPortNum portMode
        makeOneTrunkPort(nodeIndex,willowIndex,6,        0,        0,          6,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,7,        0,        0,          7,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,14,       0,        0,         14,         AC_COUPLED);
        makeOneTrunkPort(nodeIndex,willowIndex,15,       0,        0,         15,         AC_COUPLED);
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config8::makeOneWillow( int nodeIndex, int willowIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( ( (nodeIndex == 0) && (willowIndex == 0) ) ||
         ( (nodeIndex == 0) && (willowIndex == 1) ) )
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

        // Configure ganged link tabled
        makeGangedLinkTable( nodeIndex, willowIndex );
    }
    else
    {
        PRINT_ERROR("%d,%d", "Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void basicE3600Config8::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void basicE3600Config8::makeNodes()
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
            // node 0 has 4 GPUs, 2 Willow
            makeOneNode( nodeIndex, 4, 2);
            break;

        default:
            PRINT_ERROR("%d", "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void
basicE3600Config8::makeOneLwswitch( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config8::makeRemapTable( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config8::makeRIDRouteTable( int nodeIndex, int swIndex )
{
    return;
}

void
basicE3600Config8::makeRLANRouteTable( int nodeIndex, int swIndex )
{
    return;
}
#endif
