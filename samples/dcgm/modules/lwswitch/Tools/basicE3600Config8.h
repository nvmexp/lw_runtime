
#ifndef BASIC_E3600_CONFIG8_H
#define BASIC_E3600_CONFIG8_H

#include "fabricConfig.h"

//
//
//    Basic E3600 configuration 8
//
//     GPU (N0_G0)             Switch (N0_S0)                GPU (N0_G1)
//
//     N0_G0_P1    <----> N0_S0_P0       N0_S0_P8  <---->    N0_G1_P2
//     N0_G0_P0    <----> N0_S0_P1       N0_S0_P9  <---->    N0_G1_P3
//     N0_G0_P5    <----> N0_S0_P2       N0_S0_P10 <---->    N0_G1_P4
//     N0_G0_P4    <----> N0_S0_P3       N0_S0_P11 <---->    N0_G1_P5
//     N0_G0_P2    <----> N0_S0_P4       N0_S0_P12 <---->    N0_G1_P1
//     N0_G0_P3    <----> N0_S0_P5       N0_S0_P13 <---->    N0_G1_P0
// +--------------------- N0_S0_P6       N0_S0_P15 <-------------------------+
// | +------------------- N0_S0_P7       N0_S0_P14 <----------------------+  |
// | |                                   N0_S0_P16                        |  |
// | |                                   N0_S0_P17                        |  |
// | |                                                                    |  |
// | |  GPU (N0_G2)             Switch (N0_S1)                GPU (N0_G3) |  |
// | |                                                                    |  |
// | |  N0_G0_P1    <----> N0_S0_P0       N0_S0_P8  <---->    N0_G1_P2    |  |
// | |  N0_G0_P0    <----> N0_S0_P1       N0_S0_P9  <---->    N0_G1_P3    |  |
// | |  N0_G0_P5    <----> N0_S0_P2       N0_S0_P10 <---->    N0_G1_P4    |  |
// | |  N0_G0_P4    <----> N0_S0_P3       N0_S0_P11 <---->    N0_G1_P5    |  |
// | |  N0_G0_P2    <----> N0_S0_P4       N0_S0_P12 <---->    N0_G1_P1    |  |
// | |  N0_G0_P3    <----> N0_S0_P5       N0_S0_P13 <---->    N0_G1_P0    |  |
// | +-------------------> N0_S0_P7       N0_S0_P14 <---------------------+  |
// + --------------------> N0_S0_P6       N0_S0_P15 <------------------------+
//                                        N0_S0_P16
//                                        N0_S0_P17

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
//   N0_G2    2         0x0002000000000  1LL << 36       8, 9, 10, 11
//   N0_G3    3         0x07c0000000000  1LL << 36       7937, 7938, 7939, 7940
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
//    N0_S1_P0    0             0          2         1          13 (2 * 6 + 1)  ACCESS_PORT_GPU
//    N0_S1_P1    1             0          2         0          12 (2 * 6 + 0)  ACCESS_PORT_GPU
//    N0_S1_P2    2             0          2         5          17 (2 * 6 + 5)  ACCESS_PORT_GPU
//    N0_S1_P3    3             0          2         4          16 (2 * 6 + 4)  ACCESS_PORT_GPU
//    N0_S1_P4    4             0          2         2          14 (2 * 6 + 2)  ACCESS_PORT_GPU
//    N0_S1_P5    5             0          2         3          15 (2 * 6 + 3)  ACCESS_PORT_GPU
//    N0_S1_P8    8             0          3         2          20 (3 * 6 + 2)  ACCESS_PORT_GPU
//    N0_S1_P9    9             0          3         3          21 (3 * 6 + 3)  ACCESS_PORT_GPU
//    N0_S1_P10   10            0          3         4          22 (3 * 6 + 4)  ACCESS_PORT_GPU
//    N0_S1_P11   11            0          3         5          23 (3 * 6 + 5)  ACCESS_PORT_GPU
//    N0_S1_P12   12            0          3         1          19 (3 * 6 + 1)  ACCESS_PORT_GPU
//    N0_S1_P13   13            0          3         0          18 (3 * 6 + 0)  ACCESS_PORT_GPU
//
//    Trunk Ports
//    Port        localPortNum  farNodeID  farPeerID farPortNum  type
//    N0_S0_P6    6             0          1         6          TRUNK_PORT_GPU
//    N0_S0_P7    7             0          1         7          TRUNK_PORT_GPU
//    N0_S0_P14   14            0          1        14          TRUNK_PORT_GPU
//    N0_S0_P15   15            0          1        15          TRUNK_PORT_GPU
//
//    N0_S1_P6    6             0          0         6          TRUNK_PORT_GPU
//    N0_S1_P7    7             0          0         7          TRUNK_PORT_GPU
//    N0_S1_P14   14            0          0        14          TRUNK_PORT_GPU
//    N0_S1_P15   15            0          0        15          TRUNK_PORT_GPU

class basicE3600Config8 : public fabricConfig
{
public:
      basicE3600Config8( fabricTopologyEnum topo );
      virtual ~basicE3600Config8();

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

private:

};

#endif
