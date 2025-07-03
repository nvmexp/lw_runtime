#ifndef BASIC_E3600_CONFIG3_H
#define BASIC_E3600_CONFIG3_H

#include "fabricConfig.h"

#define BASIC_E3602_NUM_NODES     1
#define BASIC_E3603_NUM_WILLOWS   1
#define BASIC_E3602_NUM_GPUS      0

//
//
//    Basic E3600 configuration 3
//
//                   Switch (N0_S0)
//
//    |--------> N0_S0_P0       N0_S0_P8  <--------|
//    | |------> N0_S0_P1       N0_S0_P9  <------| |
//    | |  <---> N0_S0_P2       N0_S0_P10 <--->  | |
//    | |  <---> N0_S0_P3       N0_S0_P11 <--->  | |
//    | |------> N0_S0_P4       N0_S0_P12 <------| |
//    |--------> N0_S0_P5       N0_S0_P13 <--------|
//               N0_S0_P6       N0_S0_P15
//               N0_S0_P7       N0_S0_P14
//                              N0_S0_P16 <--->
//                              N0_S0_P17 <--->

//
//   GPU Configuration
//
//   There is no GPU in this configuration
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
//    N0_S0_P0    0             0          0         5           5 (0 * 6 + 5)  ACCESS_PORT_GPU
//    N0_S0_P1    1             0          0         4           4 (0 * 6 + 4)  ACCESS_PORT_GPU
//    N0_S0_P2    2             0          0         2           2 (0 * 6 + 2)  ACCESS_PORT_GPU
//    N0_S0_P3    3             0          0         3           3 (0 * 6 + 3)  ACCESS_PORT_GPU
//    N0_S0_P4    4             0          0         1           1 (0 * 6 + 1)  ACCESS_PORT_GPU
//    N0_S0_P5    5             0          0         0           0 (0 * 6 + 0)  ACCESS_PORT_GPU
//    N0_S0_P8    8             0          0        13          13 (0 * 6 + 13) ACCESS_PORT_GPU
//    N0_S0_P9    9             0          0        12          12 (0 * 6 + 12) ACCESS_PORT_GPU
//    N0_S0_P10   10            0          0        10          10 (0 * 6 + 10) ACCESS_PORT_GPU
//    N0_S0_P11   11            0          0        11          11 (0 * 6 + 11) ACCESS_PORT_GPU
//    N0_S0_P12   12            0          0         9           9 (0 * 6 + 9)  ACCESS_PORT_GPU
//    N0_S0_P13   13            0          0         8           8 (0 * 6 + 8)  ACCESS_PORT_GPU
//    N0_S0_P16   16            0          0        16          16 (0 * 6 + 16) ACCESS_PORT_GPU
//    N0_S0_P17   17            0          0        17          17 (0 * 6 + 17) ACCESS_PORT_GPU
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

class basicE3600Config3 : public fabricConfig
{
public:
      basicE3600Config3( fabricTopologyEnum topo );
      virtual ~basicE3600Config3();

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
