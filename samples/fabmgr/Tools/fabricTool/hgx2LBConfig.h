
#ifndef HGX2LB_CONFIG_CONFIG_H
#define HGX2LB_CONFIG_CONFIG_H

#include "explorer16common.h"
#include "fabricConfig.h"
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
//   N0_G4    4         0x0004000000000  1LL << 36       16, 17, 18, 19
//   N0_G5    5         0x0005000000000  1LL << 36       20, 21, 22, 23
//   N0_G6    6         0x0006000000000  1LL << 36       24, 25, 26, 27
//   N0_G7    7         0x0007000000000  1LL << 36       28, 29, 30, 31
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
//      GB1__GPU1    0    GB1_SW4_G4    16
//      GB1__GPU1    1    GB1_SW1_G1    5
//      GB1__GPU1    2    GB1_SW6_G6    6
//      GB1__GPU1    3    GB1_SW3_G3    15
//      GB1__GPU1    4    GB1_SW5_G5    15
//      GB1__GPU1    5    GB1_SW2_G2    6
//      GB1__GPU2    0    GB1_SW4_G4    15
//      GB1__GPU2    1    GB1_SW1_G1    16
//      GB1__GPU2    2    GB1_SW3_G3    6
//      GB1__GPU2    3    GB1_SW6_G6    12
//      GB1__GPU2    4    GB1_SW2_G2    17
//      GB1__GPU2    5    GB1_SW5_G5    7
//      GB1__GPU3    0    GB1_SW4_G4    14
//      GB1__GPU3    1    GB1_SW1_G1    17
//      GB1__GPU3    2    GB1_SW3_G3    17
//      GB1__GPU3    3    GB1_SW6_G6    13
//      GB1__GPU3    4    GB1_SW5_G5    6
//      GB1__GPU3    5    GB1_SW2_G2    4
//      GB1__GPU4    0    GB1_SW4_G4    17
//      GB1__GPU4    1    GB1_SW1_G1    4
//      GB1__GPU4    2    GB1_SW3_G3    7
//      GB1__GPU4    3    GB1_SW6_G6    7
//      GB1__GPU4    4    GB1_SW2_G2    7
//      GB1__GPU4    5    GB1_SW5_G5    17
//      GB1__GPU5    0    GB1_SW4_G4    13
//      GB1__GPU5    1    GB1_SW1_G1    13
//      GB1__GPU5    2    GB1_SW5_G5    13
//      GB1__GPU5    3    GB1_SW3_G3    13
//      GB1__GPU5    4    GB1_SW6_G6    14
//      GB1__GPU5    5    GB1_SW2_G2    16
//      GB1__GPU6    0    GB1_SW4_G4    5
//      GB1__GPU6    1    GB1_SW1_G1    14
//      GB1__GPU6    2    GB1_SW6_G6    5
//      GB1__GPU6    3    GB1_SW3_G3    4
//      GB1__GPU6    4    GB1_SW2_G2    12
//      GB1__GPU6    5    GB1_SW5_G5    14
//      GB1__GPU7    0    GB1_SW4_G4    12
//      GB1__GPU7    1    GB1_SW1_G1    15
//      GB1__GPU7    2    GB1_SW5_G5    5
//      GB1__GPU7    3    GB1_SW3_G3    5
//      GB1__GPU7    4    GB1_SW2_G2    13
//      GB1__GPU7    5    GB1_SW6_G6    17
//      GB1__GPU8    0    GB1_SW4_G4    4
//      GB1__GPU8    1    GB1_SW1_G1    12
//      GB1__GPU8    2    GB1_SW6_G6    15
//      GB1__GPU8    3    GB1_SW5_G5    12
//      GB1__GPU8    4    GB1_SW3_G3    14
//      GB1__GPU8    5    GB1_SW2_G2    5
//      GB1_SW1_G1   0    GB2_SW1_G1    3
//      GB1_SW1_G1   1    GB2_SW1_G1    2
//      GB1_SW1_G1   2    GB2_SW1_G1    1
//      GB1_SW1_G1   3    GB2_SW1_G1    0
//      GB1_SW1_G1   4    GB1__GPU4     1
//      GB1_SW1_G1   5    GB1__GPU1     1
//      GB1_SW1_G1   8    GB2_SW1_G1    11
//      GB1_SW1_G1   9    GB2_SW1_G1    10
//      GB1_SW1_G1   10   GB2_SW1_G1    9
//      GB1_SW1_G1   11   GB2_SW1_G1    8
//      GB1_SW1_G1   12   GB1__GPU8     1
//      GB1_SW1_G1   13   GB1__GPU5     1
//      GB1_SW1_G1   14   GB1__GPU6     1
//      GB1_SW1_G1   15   GB1__GPU7     1
//      GB1_SW1_G1   16   GB1__GPU2     1
//      GB1_SW1_G1   17   GB1__GPU3     1
//      GB1_SW2_G2   0    GB2_SW2_G2    3
//      GB1_SW2_G2   1    GB2_SW2_G2    2
//      GB1_SW2_G2   2    GB2_SW2_G2    1
//      GB1_SW2_G2   3    GB2_SW2_G2    0
//      GB1_SW2_G2   4    GB1__GPU3     5
//      GB1_SW2_G2   5    GB1__GPU8     5
//      GB1_SW2_G2   6    GB1__GPU1     5
//      GB1_SW2_G2   7    GB1__GPU4     4
//      GB1_SW2_G2   8    GB2_SW2_G2    11
//      GB1_SW2_G2   9    GB2_SW2_G2    10
//      GB1_SW2_G2   10   GB2_SW2_G2    9
//      GB1_SW2_G2   11   GB2_SW2_G2    8
//      GB1_SW2_G2   12   GB1__GPU6     4
//      GB1_SW2_G2   13   GB1__GPU7     4
//      GB1_SW2_G2   16   GB1__GPU5     5
//      GB1_SW2_G2   17   GB1__GPU2     4
//      GB1_SW3_G3   0    GB2_SW3_G3    3
//      GB1_SW3_G3   1    GB2_SW3_G3    2
//      GB1_SW3_G3   2    GB2_SW3_G3    1
//      GB1_SW3_G3   3    GB2_SW3_G3    0
//      GB1_SW3_G3   4    GB1__GPU6     3
//      GB1_SW3_G3   5    GB1__GPU7     3
//      GB1_SW3_G3   6    GB1__GPU2     2
//      GB1_SW3_G3   7    GB1__GPU4     2
//      GB1_SW3_G3   8    GB2_SW3_G3    11
//      GB1_SW3_G3   9    GB2_SW3_G3    10
//      GB1_SW3_G3   10   GB2_SW3_G3    9
//      GB1_SW3_G3   11   GB2_SW3_G3    8
//      GB1_SW3_G3   13   GB1__GPU5     3
//      GB1_SW3_G3   14   GB1__GPU8     4
//      GB1_SW3_G3   15   GB1__GPU1     3
//      GB1_SW3_G3   17   GB1__GPU3     2
//      GB1_SW4_G4   0    GB2_SW4_G4    3
//      GB1_SW4_G4   1    GB2_SW4_G4    2
//      GB1_SW4_G4   2    GB2_SW4_G4    1
//      GB1_SW4_G4   3    GB2_SW4_G4    0
//      GB1_SW4_G4   4    GB1__GPU8     0
//      GB1_SW4_G4   5    GB1__GPU6     0
//      GB1_SW4_G4   8    GB2_SW4_G4    11
//      GB1_SW4_G4   9    GB2_SW4_G4    10
//      GB1_SW4_G4   10   GB2_SW4_G4    9
//      GB1_SW4_G4   11   GB2_SW4_G4    8
//      GB1_SW4_G4   12   GB1__GPU7     0
//      GB1_SW4_G4   13   GB1__GPU5     0
//      GB1_SW4_G4   14   GB1__GPU3     0
//      GB1_SW4_G4   15   GB1__GPU2     0
//      GB1_SW4_G4   16   GB1__GPU1     0
//      GB1_SW4_G4   17   GB1__GPU4     0
//      GB1_SW5_G5   0    GB2_SW5_G5    3
//      GB1_SW5_G5   1    GB2_SW5_G5    2
//      GB1_SW5_G5   2    GB2_SW5_G5    1
//      GB1_SW5_G5   3    GB2_SW5_G5    0
//      GB1_SW5_G5   5    GB1__GPU7     2
//      GB1_SW5_G5   6    GB1__GPU3     4
//      GB1_SW5_G5   7    GB1__GPU2     5
//      GB1_SW5_G5   8    GB2_SW5_G5    11
//      GB1_SW5_G5   9    GB2_SW5_G5    10
//      GB1_SW5_G5   10   GB2_SW5_G5    9
//      GB1_SW5_G5   11   GB2_SW5_G5    8
//      GB1_SW5_G5   12   GB1__GPU8     3
//      GB1_SW5_G5   13   GB1__GPU5     2
//      GB1_SW5_G5   14   GB1__GPU6     5
//      GB1_SW5_G5   15   GB1__GPU1     4
//      GB1_SW5_G5   17   GB1__GPU4     5
//      GB1_SW6_G6   0    GB2_SW6_G6    3
//      GB1_SW6_G6   1    GB2_SW6_G6    2
//      GB1_SW6_G6   2    GB2_SW6_G6    1
//      GB1_SW6_G6   3    GB2_SW6_G6    0
//      GB1_SW6_G6   5    GB1__GPU6     2
//      GB1_SW6_G6   6    GB1__GPU1     2
//      GB1_SW6_G6   7    GB1__GPU4     3
//      GB1_SW6_G6   8    GB2_SW6_G6    11
//      GB1_SW6_G6   9    GB2_SW6_G6    10
//      GB1_SW6_G6   10   GB2_SW6_G6    9
//      GB1_SW6_G6   11   GB2_SW6_G6    8
//      GB1_SW6_G6   12   GB1__GPU2     3
//      GB1_SW6_G6   13   GB1__GPU3     3
//      GB1_SW6_G6   14   GB1__GPU5     4
//      GB1_SW6_G6   15   GB1__GPU8     2
//      GB1_SW6_G6   17   GB1__GPU7     5
//      GB2__GPU1    0    GB2_SW4_G4    16
//      GB2__GPU1    1    GB2_SW1_G1    5
//      GB2__GPU1    2    GB2_SW6_G6    6
//      GB2__GPU1    3    GB2_SW3_G3    15
//      GB2__GPU1    4    GB2_SW5_G5    15
//      GB2__GPU1    5    GB2_SW2_G2    6
//      GB2__GPU2    0    GB2_SW4_G4    15
//      GB2__GPU2    1    GB2_SW1_G1    16
//      GB2__GPU2    2    GB2_SW3_G3    6
//      GB2__GPU2    3    GB2_SW6_G6    12
//      GB2__GPU2    4    GB2_SW2_G2    17
//      GB2__GPU2    5    GB2_SW5_G5    7
//      GB2__GPU3    0    GB2_SW4_G4    14
//      GB2__GPU3    1    GB2_SW1_G1    17
//      GB2__GPU3    2    GB2_SW3_G3    17
//      GB2__GPU3    3    GB2_SW6_G6    13
//      GB2__GPU3    4    GB2_SW5_G5    6
//      GB2__GPU3    5    GB2_SW2_G2    4
//      GB2__GPU4    0    GB2_SW4_G4    17
//      GB2__GPU4    1    GB2_SW1_G1    4
//      GB2__GPU4    2    GB2_SW3_G3    7
//      GB2__GPU4    3    GB2_SW6_G6    7
//      GB2__GPU4    4    GB2_SW2_G2    7
//      GB2__GPU4    5    GB2_SW5_G5    17
//      GB2__GPU5    0    GB2_SW4_G4    13
//      GB2__GPU5    1    GB2_SW1_G1    13
//      GB2__GPU5    2    GB2_SW5_G5    13
//      GB2__GPU5    3    GB2_SW3_G3    13
//      GB2__GPU5    4    GB2_SW6_G6    14
//      GB2__GPU5    5    GB2_SW2_G2    16
//      GB2__GPU6    0    GB2_SW4_G4    5
//      GB2__GPU6    1    GB2_SW1_G1    14
//      GB2__GPU6    2    GB2_SW6_G6    5
//      GB2__GPU6    3    GB2_SW3_G3    4
//      GB2__GPU6    4    GB2_SW2_G2    12
//      GB2__GPU6    5    GB2_SW5_G5    14
//      GB2__GPU7    0    GB2_SW4_G4    12
//      GB2__GPU7    1    GB2_SW1_G1    15
//      GB2__GPU7    2    GB2_SW5_G5    5
//      GB2__GPU7    3    GB2_SW3_G3    5
//      GB2__GPU7    4    GB2_SW2_G2    13
//      GB2__GPU7    5    GB2_SW6_G6    17
//      GB2__GPU8    0    GB2_SW4_G4    4
//      GB2__GPU8    1    GB2_SW1_G1    12
//      GB2__GPU8    2    GB2_SW6_G6    15
//      GB2__GPU8    3    GB2_SW5_G5    12
//      GB2__GPU8    4    GB2_SW3_G3    14
//      GB2__GPU8    5    GB2_SW2_G2    5
//      GB2_SW1_G1   0    GB1_SW1_G1    3
//      GB2_SW1_G1   1    GB1_SW1_G1    2
//      GB2_SW1_G1   2    GB1_SW1_G1    1
//      GB2_SW1_G1   3    GB1_SW1_G1    0
//      GB2_SW1_G1   4    GB2__GPU4     1
//      GB2_SW1_G1   5    GB2__GPU1     1
//      GB2_SW1_G1   8    GB1_SW1_G1    11
//      GB2_SW1_G1   9    GB1_SW1_G1    10
//      GB2_SW1_G1   10   GB1_SW1_G1    9
//      GB2_SW1_G1   11   GB1_SW1_G1    8
//      GB2_SW1_G1   12   GB2__GPU8     1
//      GB2_SW1_G1   13   GB2__GPU5     1
//      GB2_SW1_G1   14   GB2__GPU6     1
//      GB2_SW1_G1   15   GB2__GPU7     1
//      GB2_SW1_G1   16   GB2__GPU2     1
//      GB2_SW1_G1   17   GB2__GPU3     1
//      GB2_SW2_G2   0    GB1_SW2_G2    3
//      GB2_SW2_G2   1    GB1_SW2_G2    2
//      GB2_SW2_G2   2    GB1_SW2_G2    1
//      GB2_SW2_G2   3    GB1_SW2_G2    0
//      GB2_SW2_G2   4    GB2__GPU3     5
//      GB2_SW2_G2   5    GB2__GPU8     5
//      GB2_SW2_G2   6    GB2__GPU1     5
//      GB2_SW2_G2   7    GB2__GPU4     4
//      GB2_SW2_G2   8    GB1_SW2_G2    11
//      GB2_SW2_G2   9    GB1_SW2_G2    10
//      GB2_SW2_G2   10   GB1_SW2_G2    9
//      GB2_SW2_G2   11   GB1_SW2_G2    8
//      GB2_SW2_G2   12   GB2__GPU6     4
//      GB2_SW2_G2   13   GB2__GPU7     4
//      GB2_SW2_G2   16   GB2__GPU5     5
//      GB2_SW2_G2   17   GB2__GPU2     4
//      GB2_SW3_G3   0    GB1_SW3_G3    3
//      GB2_SW3_G3   1    GB1_SW3_G3    2
//      GB2_SW3_G3   2    GB1_SW3_G3    1
//      GB2_SW3_G3   3    GB1_SW3_G3    0
//      GB2_SW3_G3   4    GB2__GPU6     3
//      GB2_SW3_G3   5    GB2__GPU7     3
//      GB2_SW3_G3   6    GB2__GPU2     2
//      GB2_SW3_G3   7    GB2__GPU4     2
//      GB2_SW3_G3   8    GB1_SW3_G3    11
//      GB2_SW3_G3   9    GB1_SW3_G3    10
//      GB2_SW3_G3   10   GB1_SW3_G3    9
//      GB2_SW3_G3   11   GB1_SW3_G3    8
//      GB2_SW3_G3   13   GB2__GPU5     3
//      GB2_SW3_G3   14   GB2__GPU8     4
//      GB2_SW3_G3   15   GB2__GPU1     3
//      GB2_SW3_G3   17   GB2__GPU3     2
//      GB2_SW4_G4   0    GB1_SW4_G4    3
//      GB2_SW4_G4   1    GB1_SW4_G4    2
//      GB2_SW4_G4   2    GB1_SW4_G4    1
//      GB2_SW4_G4   3    GB1_SW4_G4    0
//      GB2_SW4_G4   4    GB2__GPU8     0
//      GB2_SW4_G4   5    GB2__GPU6     0
//      GB2_SW4_G4   8    GB1_SW4_G4    11
//      GB2_SW4_G4   9    GB1_SW4_G4    10
//      GB2_SW4_G4   10   GB1_SW4_G4    9
//      GB2_SW4_G4   11   GB1_SW4_G4    8
//      GB2_SW4_G4   12   GB2__GPU7     0
//      GB2_SW4_G4   13   GB2__GPU5     0
//      GB2_SW4_G4   14   GB2__GPU3     0
//      GB2_SW4_G4   15   GB2__GPU2     0
//      GB2_SW4_G4   16   GB2__GPU1     0
//      GB2_SW4_G4   17   GB2__GPU4     0
//      GB2_SW5_G5   0    GB1_SW5_G5    3
//      GB2_SW5_G5   1    GB1_SW5_G5    2
//      GB2_SW5_G5   2    GB1_SW5_G5    1
//      GB2_SW5_G5   3    GB1_SW5_G5    0
//      GB2_SW5_G5   5    GB2__GPU7     2
//      GB2_SW5_G5   6    GB2__GPU3     4
//      GB2_SW5_G5   7    GB2__GPU2     5
//      GB2_SW5_G5   8    GB1_SW5_G5    11
//      GB2_SW5_G5   9    GB1_SW5_G5    10
//      GB2_SW5_G5   10   GB1_SW5_G5    9
//      GB2_SW5_G5   11   GB1_SW5_G5    8
//      GB2_SW5_G5   12   GB2__GPU8     3
//      GB2_SW5_G5   13   GB2__GPU5     2
//      GB2_SW5_G5   14   GB2__GPU6     5
//      GB2_SW5_G5   15   GB2__GPU1     4
//      GB2_SW5_G5   17   GB2__GPU4     5
//      GB2_SW6_G6   0    GB1_SW6_G6    3
//      GB2_SW6_G6   1    GB1_SW6_G6    2
//      GB2_SW6_G6   2    GB1_SW6_G6    1
//      GB2_SW6_G6   3    GB1_SW6_G6    0
//      GB2_SW6_G6   5    GB2__GPU6     2
//      GB2_SW6_G6   6    GB2__GPU1     2
//      GB2_SW6_G6   7    GB2__GPU4     3
//      GB2_SW6_G6   8    GB1_SW6_G6    11
//      GB2_SW6_G6   9    GB1_SW6_G6    10
//      GB2_SW6_G6   10   GB1_SW6_G6    9
//      GB2_SW6_G6   11   GB1_SW6_G6    8
//      GB2_SW6_G6   12   GB2__GPU2     3
//      GB2_SW6_G6   13   GB2__GPU3     3
//      GB2_SW6_G6   14   GB2__GPU5     4
//      GB2_SW6_G6   15   GB2__GPU8     2
//      GB2_SW6_G6   17   GB2__GPU7     5

// define enums to take hardware device ID to protobuf device ID

typedef struct HWLink16LB_t {
    unsigned int willowIndex;
    unsigned int willowPort;
    int          GPUIndex;
    unsigned int GPUPort;
} HWLink16LB;
 
class hgx2LBConfig : public fabricConfig
{
public:
      hgx2LBConfig( fabricTopologyEnum topo );
      virtual ~hgx2LBConfig();

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
      void fillSystemPartitionInfo(int nodeIndex);

};

#endif
