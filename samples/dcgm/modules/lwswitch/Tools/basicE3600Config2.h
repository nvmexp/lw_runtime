#ifndef BASIC_E3600_CONFIG2_H
#define BASIC_E3600_CONFIG2_H

#include "fabricConfig.h"

#define BASIC_E3600_NUM_NODES     1
#define BASIC_E3600_NUM_WILLOWS   1
#define BASIC_E3600_NUM_GPUS      2

//
//
//    Basic E3600 configuration 2
//
//    GPU (N0_G0)             Switch (N0_S0)                GPU (N0_G1)

//
//    N0_G0_P1    <----> N0_S0_P0       N0_S0_P8  <---->    N0_G1_P2
//    N0_G0_P0    <----> N0_S0_P1       N0_S0_P9  <---->    N0_G1_P3
//    N0_G0_P5    <----> N0_S0_P2       N0_S0_P10 <---->    N0_G1_P4
//    N0_G0_P4    <----> N0_S0_P3       N0_S0_P11 <---->    N0_G1_P5
//    N0_G0_P2    <----> N0_S0_P4       N0_S0_P12 <---->    N0_G1_P1
//    N0_G0_P3    <----> N0_S0_P5       N0_S0_P13 <---->    N0_G1_P0
//                ---->  N0_S0_P6       N0_S0_P15 <----
//                | -->  N0_S0_P7       N0_S0_P14 <-- |
//                | |                   N0_S0_P16   | |
//                | |                   N0_S0_P17   | |
//                | |                               | |
//                | |                               | |
//                | --------------------------------- |
//                -------------------------------------
//
//   Switch port 0 to 5 are connected to GPU 0, ingress packets are xbar switched to port 6 and 7.
//   Switch port 8 to 13 are connected to GPU 1, ingress packets are xbar switched to port 14 and 15.
//   Port 6, 7 and Port 15, 14 are connect to each other via external cables (looped out).
//   Port 16 and 17 are unused.
//
//   GPU Configuration
//
//   Fabric Address assignment for up to 1K GPUs
//   [46:37] top 10 bits is the GPU endpoint ID, gpuEndpointID = node index * 8 + GPU index within node.
//   Bit 36  is set to 0, so that we do not need to do any address remapping for the GPUs with 64G memory or less
//
//   fabricaddrbase  = ( gpuEndpointID << 37 );
//   fabricaddrrange = ( 1LL << 36 );
//
//   Ingress Req table Index assignment for up to 8K Entries
//   [12:3] top 10 bits is the GPU endpoint ID
//   Bit 2  is set to 0
//   [1:0]  represents the 4 16G memory region ( 4 ingress request entries).

//   IngressReqIndex = ( gpuEndpointId << 3 + 0 ), ( gpuEndpointId << 3 + 1 ),
//                     ( gpuEndpointId << 3 + 2 ), ( gpuEndpointId << 3 + 3 )
//
//   In bringup fabricaddrbase is hardcoded
//
//   GPU    endpointID  fabricAddrBase  fabricAddrRange  IngressReqIndex
//   N0_G0    0         0x1ab0000000000  1LL << 36       1708, 1709, 1710, 1711
//   N0_G1    1         0x02ac000000000  1LL << 36       171, 172, 173, 174
//
//    Switch Configuration
//
//    Port Configuration
//
//    Access Ports
//
//    All entries accessPort.switchPortConfig.type = ACCESS_PORT_GPU
//    RequesterLinkID = enpointID * 6 + endpoint port index
//    farPeerID is the endpointID of GPU/NPU that is connected to the switch
//
//    Port        localPortNum  farNodeID  farPeerID farPortNum  RequesterLinkID type
//    N0_S0_P0    0             0          0         1           1 (0 * 6 + 1)  ACCESS_PORT_GPU
//    N0_S0_P1    1             0          0         0           0 (0 * 6 + 0)  ACCESS_PORT_GPU
//    N0_S0_P2    2             0          0         5           5 (0 * 6 + 5)  ACCESS_PORT_GPU
//    N0_S0_P3    3             0          0         4           4 (0 * 6 + 4)  ACCESS_PORT_GPU
//    N0_S0_P4    4             0          0         2           2 (0 * 6 + 2)  ACCESS_PORT_GPU
//    N0_S0_P5    5             0          0         3           3 (0 * 6 + 3)  ACCESS_PORT_GPU
//    N0_S0_P8    8             0          1         2           8 (1 * 6 + 2)  ACCESS_PORT_GPU
//    N0_S0_P9    9             0          1         3           9 (1 * 6 + 3)  ACCESS_PORT_GPU
//    N0_S0_P10   10            0          1         4          10 (1 * 6 + 4)  ACCESS_PORT_GPU
//    N0_S0_P11   11            0          1         5          11 (1 * 6 + 5)  ACCESS_PORT_GPU
//    N0_S0_P12   12            0          1         1           7 (1 * 6 + 1)  ACCESS_PORT_GPU
//    N0_S0_P13   13            0          1         0           6 (1 * 6 + 0)  ACCESS_PORT_GPU
//
//    Trunk Ports
//    All entries trunkPort.switchPortConfig.type = TRUNK_PORT_SWITCH
//
//    Port        localPortNum  farNodeID  farSwitchID farPortNum type
//    N0_S0_P6    6             0          0           15         TRUNK_PORT_SWITCH
//    N0_S0_P7    7             0          0           14         TRUNK_PORT_SWITCH
//    N0_S0_P14   14            0          0           7          TRUNK_PORT_SWITCH
//    N0_S0_P15   15            0          0           6          TRUNK_PORT_SWITCH
//
//
//    Ingress Request and Response table
//
//    Access port N0_S0_P0 to N0_S0_P5
//
//    Ingress Req Table
//    Ingress req going to N0_G1 (fabric address [46:34] 8 to 11)
//    will be switched to trunk port N0_S0_P6 and N0_S0_P7 VC mode 0
//    index        address routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//      171 0x02ac00000000           0     0x11000000      0x00000000       0x00000000          1
//      172 0x02b000000000           0     0x11000000      0x00000000       0x00000000          1
//      173 0x02b400000000           0     0x11000000      0x00000000       0x00000000          1
//      174 0x02b800000000           0     0x11000000      0x00000000       0x00000000          1
//
//    Ingress Resp Table
//    Ingress resp going to N0_G1 (RequesterLinkID 6 to 11)
//    will be switched to trunk port N0_S0_P6 and N0_S0_P7 VC mode 0
//    index routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//        6           0     0x11000000      0x00000000       0x00000000          1
//        7           0     0x11000000      0x00000000       0x00000000          1
//        8           0     0x11000000      0x00000000       0x00000000          1
//        9           0     0x11000000      0x00000000       0x00000000          1
//       10           0     0x11000000      0x00000000       0x00000000          1
//       11           0     0x11000000      0x00000000       0x00000000          1
//
//
//    Access N0_S0_P8 to N0_S0_P13
//
//    Ingress Req Table
//    Ingress req going to N0_G0 (fabric address [46:34] 0 to 3)
//    will be switched to trunk port N0_S0_P14 and N0_S0_P15 VC mode 0
//    index        address routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//     1708 0x1ab000000000           0     0x00000000      0x11000000       0x00000000          1
//     1709 0x1ab400000000           0     0x00000000      0x11000000       0x00000000          1
//     1710 0x1ab800000000           0     0x00000000      0x11000000       0x00000000          1
//     1711 0x1abc00000000           0     0x00000000      0x11000000       0x00000000          1
//
//    Ingress Resp Table
//    Ingress resp going to N0_G0 (RequesterLinkID 0 to 5)
//    will be switched to trunk port N0_S0_P14 and N0_S0_P15 VC mode 0
//    index routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//        0           0     0x00000000      0x11000000       0x00000000          1
//        1           0     0x00000000      0x11000000       0x00000000          1
//        2           0     0x00000000      0x11000000       0x00000000          1
//        3           0     0x00000000      0x11000000       0x00000000          1
//        4           0     0x00000000      0x11000000       0x00000000          1
//        5           0     0x00000000      0x11000000       0x00000000          1
//
//
//    Trunk N0_S0_P6 to N0_S0_P7
//
//    Ingress Req Table
//    Ingress req going to N0_G0 (fabric address [46:34] 0 to 3)
//    will be switched to access port N0_S0_P0 to N0_S0_P5 VC mode 0
//    index        address routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//     1708 0x1ab000000000           0     0x00111111      0x00000000       0x00000000          1
//     1709 0x1ab400000000           0     0x00111111      0x00000000       0x00000000          1
//     1710 0x1ab800000000           0     0x00111111      0x00000000       0x00000000          1
//     1711 0x1abc00000000           0     0x00111111      0x00000000       0x00000000          1
//
//    Ingress Resp Table
//    Ingress resp going to N0_G0 (RequesterLinkID 0 to 5)
//    will be switched to access port N0_S0_P0 to N0_S0_P5 VC mode 0
//    index routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//        1           0     0x00000001      0x00000000       0x00000000          1
//        0           0     0x00000010      0x00000000       0x00000000          1
//        5           0     0x00000100      0x00000000       0x00000000          1
//        4           0     0x00001000      0x00000000       0x00000000          1
//        2           0     0x00010000      0x00000000       0x00000000          1
//        3           0     0x00100000      0x00000000       0x00000000          1
//
//
//    Trunk port N0_S0_P14 to N0_S0_P15
//
//    Ingress Req Table (fabric address [46:34] 8 to 11)
//    Ingress req going to N0_G1
//    will be switched to access port N0_S0_P8 to N0_S0_P13 VC mode 0
//    index        address routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//      171 0x02ac00000000           0     0x00000000      0x00111111       0x00000000          1
//      172 0x02b000000000           0     0x00000000      0x00111111       0x00000000          1
//      173 0x02b400000000           0     0x00000000      0x00111111       0x00000000          1
//      174 0x02b800000000           0     0x00000000      0x00111111       0x00000000          1
//
//    Ingress Resp Table
//    Ingress resp going to N0_G1 (RequesterLinkID 6 to 11)
//    will be switched to access port N0_S0_P8 to N0_S0_P13 VC mode 0
//    index routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//        8           0     0x00000000      0x00000001       0x00000000          1
//        9           0     0x00000000      0x00000010       0x00000000          1
//       10           0     0x00000000      0x00000100       0x00000000          1
//       11           0     0x00000000      0x00001000       0x00000000          1
//        7           0     0x00000000      0x00010000       0x00000000          1
//        6           0     0x00000000      0x00100000       0x00000000          1


class basicE3600Config2 : public fabricConfig
{
public:
      basicE3600Config2( fabricTopologyEnum topo );
      virtual ~basicE3600Config2();

      virtual void    makeOneNode( int nodeIndex, int gpuNum, int willowNum );
                                              // make one node, and add the GPUs, NPUs and Willows
      virtual void    makeNodes();            // make the nodes

      virtual void    makeOneWillow( int nodeIndex, int willowIndex );

      virtual void makeIngressReqTable( int nodeIndex, int willowIndex );

      virtual void makeIngressRespTable( int nodeIndex, int willowIndex );

      virtual void makeGangedLinkTable( int nodeIndex, int willowIndex );

      virtual void makeAccessPorts( int nodeIndex, int willowIndex );

      virtual void makeTrunkPorts( int nodeIndex, int willowIndex );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
      virtual void makeOneLwswitch( int nodeIndex, int swIndex );
      virtual void makeRemapTable( int nodeIndex, int swIndex );
      virtual void makeRIDRouteTable( int nodeIndex, int swIndex );
      virtual void makeRLANRouteTable( int nodeIndex, int swIndex );
#endif
};

#endif
