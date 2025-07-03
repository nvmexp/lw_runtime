
#ifndef EMULATION_CONFIG_H
#define EMULATION_CONFIG_H

#include "fabricConfig.h"

#define EMULATION_NUM_NODES     1
#define EMULATION_NUM_WILLOWS   1
#define EMULATION_NUM_GPUS      1

//
//
//    Basic E3600 configuration 1
//
//    GPU (N0_G0)             Switch (N0_S0)                GPU (N0_G1)
//
//                       N0_S0_P0       N0_S0_P8
//                       N0_S0_P1       N0_S0_P9
//                       N0_S0_P2       N0_S0_P10
//                       N0_S0_P3       N0_S0_P11
//                       N0_S0_P4       N0_S0_P12
//                       N0_S0_P5       N0_S0_P13
//                       N0_S0_P6       N0_S0_P14
//                       N0_S0_P7       N0_S0_P15
//    N0_G0_P0  <----->  N0_S0_P16      N0_S0_P17
//
//    traffic from switch port P16 RX will go out P16 TX
//
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
//   N0_G0    0         0x1ab0000000000  1LL << 35       1708, 1709
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
//    N0_S0_P16   16            0          0         0           0 (0 * 6 + 0)  ACCESS_PORT_GPU
//
//
//    Ingress Request and Response table
//
//    Access port N0_S0_P0 to N0_S0_P5
//
//    Ingress Req Table
//    Access N0_S0_P16 to N0_S0_P16
//
//    Ingress Req Table
//    Ingress req going to N0_G0 (fabric address [46:34] 0 to 3)
//    will be switched to trunk port N0_S0_P0 and N0_S0_P5 VC mode 0
//    Index address        routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//     1708 0x1ab000000000           0     0x00000000      0x00000000       0x00000001          1
//     1709 0x1ab400000000           0     0x00000000      0x00000000       0x00000001          1
//
//    Ingress Resp Table
//    Ingress resp going to N0_G0 (RequesterLinkID 0 to 5)
//    will be switched to trunk port N0_S0_P0 to N0_S0_P5 VC mode 0
//    Index routePolicy vcModeValid7_0 vcModeValid15_8 vcModeValid17_16 entryValid
//        0           0     0x00000000      0x00000000       0x00000001          1
//

class emulationConfig : public fabricConfig
{
public:
      emulationConfig( fabricTopologyEnum topo );
      virtual ~emulationConfig();

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
