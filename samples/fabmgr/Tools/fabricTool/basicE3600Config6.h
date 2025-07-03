
#ifndef BASIC_E3600_CONFIG6_H
#define BASIC_E3600_CONFIG6_H

#include "fabricConfig.h"

//
//
//    Basic E3600 configuration 6
//
//    GPU (N0_G0)             Switch (N0_S0)                Switch (N0_S1)
//
//    N0_G0_P1    <----> N0_S0_P0       N0_S0_P8
//    N0_G0_P0    <----> N0_S0_P1       N0_S0_P9
//    N0_G0_P5    <----> N0_S0_P2       N0_S0_P10
//    N0_G0_P4    <----> N0_S0_P3       N0_S0_P11
//    N0_G0_P2    <----> N0_S0_P4       N0_S0_P12
//    N0_G0_P3    <----> N0_S0_P5       N0_S0_P13
//                       N0_S0_P6       N0_S0_P14 <---->    N0_S1_P7
//                       N0_S0_P7       N0_S0_P15 <---->    N0_S1_P6
//                                      N0_S0_P16
//                                      N0_S0_P17
//   switch N0_S0
//   Switch port 0 to 5 are connected to GPU 0, ingress packets are xbar switched to port 14 and 15
//
//   switch N0_S1
//   ingress packets are xbar switched to port 6 and 7
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
//
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
//

class basicE3600Config6 : public fabricConfig
{
public:
      basicE3600Config6( fabricTopologyEnum topo );
      virtual ~basicE3600Config6();

      virtual void    makeOneNode( int nodeIndex, int gpuNum, int willowNum );
                                              // make one node, and add the GPUs, NPUs and Willows
      virtual void    makeNodes();            // make the nodes

      virtual void    makeOneWillow( int nodeIndex, int willowIndex );

      virtual void makeIngressReqTable( int nodeIndex, int willowIndex );

      virtual void makeIngressRespTable( int nodeIndex, int willowIndex );

      virtual void makeGangedLinkTable( int nodeIndex, int willowIndex );

      virtual void makeAccessPorts( int nodeIndex, int willowIndex );

      virtual void makeTrunkPorts( int nodeIndex, int willowIndex );
private:

};

#endif
