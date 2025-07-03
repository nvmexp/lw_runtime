
#ifndef EXPLORER_CONFIG_H
#define EXPLORER_CONFIG_H

#include "fabricConfig.h"
#include "explorer8common.h"
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
//   N0_G0    0         0x0000000000000  1LL << 36        0,  1,  2,  3
//   N0_G1    1         0x0001000000000  1LL << 36        4,  5,  6,  7
//   N0_G2    2         0x0002000000000  1LL << 36        8,  9, 10, 11
//   N0_G3    3         0x0003000000000  1LL << 36       12, 13, 14, 15
//   N0_G3    4         0x0004000000000  1LL << 36       16, 17, 18, 19
//   N0_G3    5         0x0005000000000  1LL << 36       20, 21, 22, 23
//   N0_G3    6         0x0006000000000  1LL << 36       24, 25, 26, 27
//   N0_G3    7         0x0007000000000  1LL << 36       28, 29, 30, 31
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
//
//  The following switch and port connectivity table was derived from the hardware
//  design provided by the manufacturer. Note switch and GPU indexing start with 1, not 0
//
//    Switch    Port    GPU         Port
//    U1LWP1	0	    LWL_GPU1	1
//    U1LWP1	1	    LWL_GPU1	0
//    U1LWP1	2	    LWL_GPU8	3
//    U1LWP1	3	    LWL_GPU8	2
//    U1LWP1	4	    LWL_GPU4	0
//    U1LWP1	5	    LWL_GPU4	1
//    U1LWP1	6	    LWL_GPU5	2
//    U1LWP1	7	    LWL_GPU5	3
//    U1LWP1	8	    LWL_GPU3	0
//    U1LWP1	9	    LWL_GPU3	1
//    U1LWP1	10	    LWL_GPU6	2
//    U1LWP1	11	    LWL_GPU6	3
//    U1LWP1	12	    LWL_GPU2	1
//    U1LWP1	13	    LWL_GPU2	0
//    U1LWP1	14	    LWL_GPU7	3
//    U1LWP1	15	    LWL_GPU7	2
//    U1LWP2	0	    LWL_GPU5	5
//    U1LWP2	1	    LWL_GPU5	4
//    U1LWP2	2	    LWL_GPU7	5
//    U1LWP2	3	    LWL_GPU7	4
//    U1LWP2	4	    LWL_GPU3	4
//    U1LWP2	5	    LWL_GPU3	5
//    U1LWP2	6	    LWL_GPU1	4
//    U1LWP2	7	    LWL_GPU1	5
//    U1LWP2	8	    LWL_GPU6	4
//    U1LWP2	9	    LWL_GPU6	5
//    U1LWP2	10	    LWL_GPU8	4
//    U1LWP2	11	    LWL_GPU8	5
//    U1LWP2	12	    LWL_GPU4	5
//    U1LWP2	13	    LWL_GPU4	4
//    U1LWP2	14	    LWL_GPU2	5
//    U1LWP2	15	    LWL_GPU2	4
//    U1LWP3	0	    LWL_GPU7	1
//    U1LWP3	1	    LWL_GPU7	0
//    U1LWP3	2	    LWL_GPU2	3
//    U1LWP3	3	    LWL_GPU2	2
//    U1LWP3	4	    LWL_GPU6	0
//    U1LWP3	5	    LWL_GPU6	1
//    U1LWP3	6	    LWL_GPU3	2
//    U1LWP3	7	    LWL_GPU3	3
//    U1LWP3	8	    LWL_GPU5	0
//    U1LWP3	9	    LWL_GPU5	1
//    U1LWP3	10	    LWL_GPU4	2
//    U1LWP3	11	    LWL_GPU4	3
//    U1LWP3	12	    LWL_GPU8	1
//    U1LWP3	13	    LWL_GPU8	0
//    U1LWP3	14	    LWL_GPU1	3
//    U1LWP3	15	    LWL_GPU1	2

class explorerConfig : public fabricConfig
{
public:
      explorerConfig( fabricTopologyEnum topo );
      virtual ~explorerConfig();

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
