#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <string.h>

#include "vanguard.h"
#include "fm_log.h"

vanguardConfig::vanguardConfig( fabricTopologyEnum topo ) : fabricConfig( topo )
{
    for (uint64_t i = 0; i < 16; i++)
    {

        gpuFabricAddrBase[i]  = i << 36;
        gpuFabricAddrRange[i] = FAB_ADDR_RANGE_16G * 2;
    }    
};

vanguardConfig::~vanguardConfig()
{
    if ( mFabric )
    {
        delete( mFabric );
        mFabric = NULL;
    }
};

void vanguardConfig::makeIngressReqTable( int nodeIndex, int willowIndex )
{
    int32_t gpuEndpointId, portIndex, index, i;
    int64_t mappedAddr = 0, range;
    GPU *gpu;
    int localPeer0; // connect to access port 8 to 13
    int localPeer1; // connect to access port 0 to 5
    int peerId;

    if ( nodeIndex != 0 )
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
        return;
    }

    switch ( willowIndex )
    {
        case 0:
        case 1:
        case 2:
        case 3:
            localPeer0 = 2 * willowIndex;
            localPeer1 = 2 * willowIndex + 1;
            break;

        case 4:
            localPeer0 = 10;
            localPeer1 = 11;
            break;

        case 5:
            localPeer0 = 8;
            localPeer1 = 9;
            break;

        case 6:
            localPeer0 = 14;
            localPeer1 = 15;
            break;

        case 7:
            localPeer0 = 13;
            localPeer1 = 12;
            break;

        default:
            FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
            return;
    }

    // Access port 0 to 5
    for ( portIndex = 0; portIndex <= 5; portIndex++ )
    {
        // To farPeer0, directly connected to port 8 to 13
        gpu   = gpus[nodeIndex][localPeer0];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    0x00000000,    // vcModeValid7_0
                                    1 << (4*portIndex),   // vcModeValid15_8, port 8 to 13
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            gpu   = gpus[nodeIndex][peerId];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                if ( ( willowIndex == 0 ) || ( willowIndex == 2 )  ||
                     ( willowIndex == 7 ) || ( willowIndex == 5 ))
                {
                    // use trunk port 6 and 7
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x01000000,    // vcModeValid7_0, port 6 and 7
                                            0x00000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                }
                else
                {
                    // use trunk port 14 and 15
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x00000000,    // vcModeValid7_0, port 6 and 7
                                            0x01000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                }
            }
        }
    } // Access port 0 to 5

    // Access port 8 to 13
    for ( portIndex = 8; portIndex <= 13; portIndex++ )
    {
        // To farPeer1, directly connected to port 0 to 5
        gpu   = gpus[nodeIndex][localPeer1];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    1 << (4*(portIndex - 8)),    // vcModeValid7_0, port 0 to 5
                                    0x00000000,    // vcModeValid15_8
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            gpu   = gpus[nodeIndex][peerId];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                if ( ( willowIndex == 0 ) || ( willowIndex == 2 )  ||
                     ( willowIndex == 7 ) || ( willowIndex == 5 ))
                {
                    // use trunk port 6 and 7
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x01000000,    // vcModeValid7_0, port 6 and 7
                                            0x00000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                }
                else
                {
                    // use trunk port 14 and 15
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x00000000,    // vcModeValid7_0, port 6 and 7
                                            0x01000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                }
            }
        }
    } // Access port 8 to 13

    // Trunk port 6, 7
    for ( portIndex = 6; portIndex <= 7; portIndex++ )
    {
        // To localPeer0, directly connected to port 8 to 13
        gpu   = gpus[nodeIndex][localPeer0];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    0x00000000,    // vcModeValid7_0
                                    0x00555555,    // vcModeValid15_8, port 8 to 13, force to VC set 0 $$$
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To localPeer1, directly connected to port 0 to 5
        gpu   = gpus[nodeIndex][localPeer1];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    0x00555555,    // vcModeValid7_0, port 0 to 5, force to VC set 0 $$$
                                    0x00000000,    // vcModeValid15_8
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            gpu   = gpus[nodeIndex][peerId];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                // use trunk port 14 and 15
                makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                        willowIndex,   // willowIndex
                                        portIndex,     // portIndex
                                        index,         // ingress req table index
                                        mappedAddr,    // 64 bits fabric address
                                        0,             // routePolicy
                                        0x00000000,    // vcModeValid7_0, port 6 and 7
                                        0x01000000,    // vcModeValid15_8, port 14 and 15
                                        0x00000000,    // vcModeValid17_16
                                        1);            // entryValid
            }
        }
    } // Trunk port 6, 7

    // Trunk port 14, 15
    for ( portIndex = 14; portIndex <= 15; portIndex++ )
    {
        // To localPeer0, directly connected to port 8 to 13
        gpu   = gpus[nodeIndex][localPeer0];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    0x00000000,    // vcModeValid7_0
                                    0x00555555,    // vcModeValid15_8, port 8 to 13, force to VC set 0 $$$
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To localPeer1, directly connected to port 0 to 5
        gpu   = gpus[nodeIndex][localPeer1];
        index = gpu->fabricaddrbase() >> 34;
        range = gpu->fabricaddrrange();

        for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
        {
            if ( range <= (FAB_ADDR_RANGE_16G * 4) )
            {
                mappedAddr = 0;
            }

            makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                    willowIndex,   // willowIndex
                                    portIndex,     // portIndex
                                    index,         // ingress req table index
                                    mappedAddr,    // 64 bits fabric address
                                    0,             // routePolicy
                                    0x00555555,    // vcModeValid7_0, port 0 to 5, force to VC set 0 $$$
                                    0x00000000,    // vcModeValid15_8
                                    0x00000000,    // vcModeValid17_16
                                    1);            // entryValid
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            gpu   = gpus[nodeIndex][peerId];
            index = gpu->fabricaddrbase() >> 34;
            range = gpu->fabricaddrrange();

            for ( i = 0; i < NUM_INGR_REQ_ENTRIES_PER_GPU; i++, index++)
            {
                if ( range <= (FAB_ADDR_RANGE_16G * 4) )
                {
                    mappedAddr = ( (int64_t)index ) << 34;
                }

                // use trunk port 14 and 15
                // $$$ Force VC Flip for switch 0 	
                if ( willowIndex == 0 ) {
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x03000000,    // vcModeValid7_0, port 6 and 7
                                            0x00000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                } else {
                    makeOneIngressReqEntry( nodeIndex,     // nodeIndex
                                            willowIndex,   // willowIndex
                                            portIndex,     // portIndex
                                            index,         // ingress req table index
                                            mappedAddr,    // 64 bits fabric address
                                            0,             // routePolicy
                                            0x01000000,    // vcModeValid7_0, port 6 and 7
                                            0x00000000,    // vcModeValid15_8, port 14 and 15
                                            0x00000000,    // vcModeValid17_16
                                            1);            // entryValid
                }
            }
        }
    } // Trunk port 14, 15
}

void vanguardConfig::makeIngressRespTable( int nodeIndex, int willowIndex )
{
    int gpuIndex, portIndex, i, index, peerId, outPortNum;
    accessPort *outPort;
    int localPeer0; // connect to access port 8 to 13
    int localPeer1; // connect to access port 0 to 5

    if ( nodeIndex != 0 )
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
        return;
    }

    switch ( willowIndex )
    {
        case 0:
        case 1:
        case 2:
        case 3:
            localPeer0 = 2 * willowIndex;
            localPeer1 = 2 * willowIndex + 1;
            break;

        case 4:
            localPeer0 = 10;
            localPeer1 = 11;
            break;

        case 5:
            localPeer0 = 8;
            localPeer1 = 9;
            break;

        case 6:
            localPeer0 = 14;
            localPeer1 = 15;
            break;

        case 7:
            localPeer0 = 13;
            localPeer1 = 12;
            break;

        default:
            FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
            return;
    }

    // Access port 0 to 5
    for ( portIndex = 0; portIndex <= 5; portIndex++ )
    {
        // localPeer0 is connected to outPort access port 8 to 13
        // index is the requesterLinkId of localPeer0
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
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            for ( index = (peerId*6); index <= ((peerId*6)+5); index++ )
            {
                if ( ( willowIndex == 0 ) || ( willowIndex == 2 )  ||
                     ( willowIndex == 7 ) || ( willowIndex == 5 ))
                {
                    // use trunk port 6 and 7
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x10000000,    // vcModeValid7_0, port 6 and 7
                                             0x00000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                }
                else
                {
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0, port 6 and 7
                                             0x10000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                }
            }
        }
    } // Access port 0 to 5

    // Access port 8 to 13
    for ( portIndex = 8; portIndex <= 13; portIndex++ )
    {
        // localPeer1 is connected to outPort access port 0 to 5
        // index is the requesterLinkId of localPeer1
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
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            for ( index = (peerId*6); index <= ((peerId*6)+5); index++ )
            {
                if ( ( willowIndex == 0 ) || ( willowIndex == 2 )  ||
                     ( willowIndex == 7 ) || ( willowIndex == 5 ))
                {
                    // use trunk port 6 and 7
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x10000000,    // vcModeValid7_0, port 6 and 7
                                             0x00000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                }
                else
                {
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x00000000,    // vcModeValid7_0, port 6 and 7
                                             0x10000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                }
            }
        }

    } // Access port 8 to 13

    // Trunk port 6, 7
    for ( portIndex = 6; portIndex <= 7; portIndex++ )
    {
        // localPeer0 is connected to outPort access port 8 to 13
        // index is the requesterLinkId of localPeer0
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
                                         5 << (4*(outPortNum - 8)), // vcModeValid15_8, force to VC set 0 $$$
                                         0x00000000,    // vcModeValid17_16
                                         1);            // entryValid
            } else
            {
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // localPeer1 is connected to outPort access port 0 to 5
        // index is the requesterLinkId of localPeer1
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
                                         5 << (4*outPortNum), // vcModeValid7_0, force to VC set 0 $$$
                                         0x00000000,    // vcModeValid15_8
                                         0x00000000,    // vcModeValid17_16
                                         1);            // entryValid
            } else
            {
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            for ( index = (peerId*6); index <= ((peerId*6)+5); index++ )
            {
                // use trunk port 14 and 15
                makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                         willowIndex,   // willowIndex
                                         portIndex,     // portIndex
                                         index,         // Ingress resq table index
                                         0,             // routePolicy
                                         0x00000000,    // vcModeValid7_0, port 6 and 7
                                         0x10000000,    // vcModeValid15_8, port 14 and 15
                                         0x00000000,    // vcModeValid17_16
                                         1);            // entryValid
            }
        }
    } // Trunk port 6, 7

    // Trunk port 14, 15
    for ( portIndex = 14; portIndex <= 15; portIndex++ )
    {
        // localPeer0 is connected to outPort access port 8 to 13
        // index is the requesterLinkId of localPeer0
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
                                         5 << (4*(outPortNum - 8)), // vcModeValid15_8, force to VC set 0 $$$
                                         0x00000000,    // vcModeValid17_16
                                         1);            // entryValid
            } else
            {
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // localPeer1 is connected to outPort access port 0 to 5
        // index is the requesterLinkId of localPeer1
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
                                         5 << (4*outPortNum), // vcModeValid7_0, force to VC set 0 $$$
                                         0x00000000,    // vcModeValid15_8
                                         0x00000000,    // vcModeValid17_16
                                         1);            // entryValid
            } else
            {
                FM_LOG_ERROR("invalid out going port nodeIndex %d, willowIndex %d, portIndex %d.\n",
                            nodeIndex, willowIndex, outPortNum);
            }
        }

        // To all other GPUs
        for ( peerId = 0; peerId < 16; peerId++ )
        {
            if ( ( peerId == localPeer0 ) || ( peerId == localPeer1 ) )
                continue;

            for ( index = (peerId*6); index <= ((peerId*6)+5); index++ )
            {
                // use trunk port 6 and 7
                // $$$ Force VC Flip for switch 0 	
                if ( willowIndex == 0 ) {
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x30000000,    // vcModeValid7_0, port 6 and 7
                                             0x00000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                } else {
                    makeOneIngressRespEntry( nodeIndex,     // nodeIndex
                                             willowIndex,   // willowIndex
                                             portIndex,     // portIndex
                                             index,         // Ingress resq table index
                                             0,             // routePolicy
                                             0x10000000,    // vcModeValid7_0, port 6 and 7
                                             0x00000000,    // vcModeValid15_8, port 14 and 15
                                             0x00000000,    // vcModeValid17_16
                                             1);            // entryValid
                }
            }
        }
    } // Trunk port 14, 15
}

void vanguardConfig::makeGangedLinkTable( int nodeIndex, int willowIndex )
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


void vanguardConfig::makeAccessPorts( int nodeIndex, int willowIndex )
{
    int farPeer0; // connect to access port 8 to 13
    int farPeer1; // connect to access port 0 to 5

    if ( nodeIndex != 0 )
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
        return;
    }

    switch ( willowIndex )
    {
        case 0:
        case 1:
        case 2:
        case 3:
            farPeer0 = 2 * willowIndex;
            farPeer1 = 2 * willowIndex + 1;
            break;

        case 4:
            farPeer0 = 10;
            farPeer1 = 11;
            break;

        case 5:
            farPeer0 = 8;
            farPeer1 = 9;
            break;

        case 6:
            farPeer0 = 14;
            farPeer1 = 15;
            break;

        case 7:
            farPeer0 = 13;
            farPeer1 = 12;
            break;

        default:
            FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
            return;
    }

    //                nodeIndex willowIndex   portIndex farNodeID farPeerID farPortNum portMode
    makeOneAccessPort(0,        willowIndex,  0,        0,        farPeer1, 1,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  1,        0,        farPeer1, 0,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  2,        0,        farPeer1, 5,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  3,        0,        farPeer1, 4,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  4,        0,        farPeer1, 2,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  5,        0,        farPeer1, 3,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  8,        0,        farPeer0, 2,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  9,        0,        farPeer0, 3,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  10,       0,        farPeer0, 4,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  11,       0,        farPeer0, 5,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  12,       0,        farPeer0, 1,         DC_COUPLED);
    makeOneAccessPort(0,        willowIndex,  13,       0,        farPeer0, 0,         DC_COUPLED);
}

void vanguardConfig::makeTrunkPorts( int nodeIndex, int willowIndex )
{
    if ( nodeIndex != 0 )
    {
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
        return;
    }

    switch ( willowIndex )
    {
        case 0:	//$$$Hack for now. VC Set 1 parameter is encoded in high bits of portIndex

            //               nodeIndex willowIndex   portIndex   farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,       0,        6,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,       0,        6,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        4,          14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        4,          15,         AC_COUPLED);
            break;

        case 1:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        7,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        7,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        5,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        5,           15,         AC_COUPLED);
            break;

        case 2:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        3,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        3,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        6,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        6,           15,         AC_COUPLED);
            break; 

        case 3:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        2,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        2,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        7,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        7,           15,         AC_COUPLED);
            break;

        case 4:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        5,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        5,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        0,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        0,           15,         AC_COUPLED);
            break;

        case 5:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        4,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        4,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        1,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        1,           15,         AC_COUPLED);
            break;

        case 6:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        0,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        0,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        2,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        2,           15,         AC_COUPLED);
            break;

        case 7:
            //               nodeIndex willowIndex   portIndex farNodeID farSwitchID  farPortNum portMode
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040006,        0,        1,           6,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x00040007,        0,        1,           7,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000E,       0,        3,           14,         AC_COUPLED);
            makeOneTrunkPort(nodeIndex,willowIndex,0x0004000F,       0,        3,           15,         AC_COUPLED);
            break;

        default:
            FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
            return;
    }
}

void vanguardConfig::makeOneWillow( int nodeIndex, int willowIndex )
{
    std::stringstream ecid;

    ecid << "N" << nodeIndex << "_S" << willowIndex;

    if ( (nodeIndex == 0) && (willowIndex < 8) )
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
        FM_LOG_ERROR("Invalid Willow nodeIndex %d willowIndex %d.\n", nodeIndex, willowIndex);
    }
}

void vanguardConfig::makeOneNode( int nodeIndex, int gpuNum, int willowNum )
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

void vanguardConfig::makeNodes()
{
    int i, nodeIndex = 0;
    std::stringstream nodeip;
    
    // Vanguard is single node
    for (nodeIndex = 0; nodeIndex < 1; nodeIndex++)
    {
        nodes[nodeIndex] = mFabric->add_fabricnode();
        nodes[nodeIndex]->set_version( FABRIC_MANAGER_VERSION );

        // set up node IP address
        //nodeip << "192.168.0." << (nodeIndex + 1);
        //nodes[nodeIndex]->set_ipaddress( nodeip.str().c_str() );

        switch ( nodeIndex ) {
        case 0:
            // node 0 has 16 GPUs, 8 Willow
            makeOneNode( nodeIndex, 16, 8);
            break;

        default:
            FM_LOG_ERROR( "Invalid nodeIndex %d.\n", nodeIndex);
            break;
        }
    }
}

