
#ifndef BASIC_E4700_CONFIG1_H
#define BASIC_E4700_CONFIG1_H

#include "fabricConfig.h"

class basicE4700Config1 : public fabricConfig
{
public:
    typedef enum {
        NotPopulated  = 0,
        PG506,       // GA100 in SXM4 slot
        E4702,       // Loop back in SXM4 slot
        E4705        // Loop back in ExaMAX slot
    } E4700EndpointType_t;

      basicE4700Config1( fabricTopologyEnum topo );

      virtual ~basicE4700Config1();

      virtual void makeOneNode( int nodeIndex, int gpuNum, int lrNum );
                                           // make one node, and add the GPUs, NPUs and LRs
      virtual void makeNodes();            // make the nodes

      virtual void makeOneLwswitch( int nodeIndex, int swIndex );

      virtual void makeRemapTable( int nodeIndex, int swIndex );

      virtual void makeRIDRouteTable( int nodeIndex, int swIndex );

      virtual void makeRLANRouteTable( int nodeIndex, int swIndex );

      virtual void makeRIDandRlanRouteTable( int nodeIndex, int swIndex );

      virtual void makeGangedLinkTable( int nodeIndex, int swIndex );

      virtual void makeAccessPorts( int nodeIndex, int swIndex );

      virtual void makeTrunkPorts( int nodeIndex, int swIndex );

      virtual void  makeOneWillow( int nodeIndex, int swIndex );

      virtual void makeIngressReqTable( int nodeIndex, int swIndex );

      virtual void makeIngressRespTable( int nodeIndex, int swIndex );

      void setE4700Config(E4700EndpointType_t sxm4Slot0, E4700EndpointType_t sxm4Slot1,
                          E4700EndpointType_t exaMAXslot0, E4700EndpointType_t exaMAXslot1,
                          bool enableWarmup, bool enableSpray, bool useTrunkPort);

private:
      typedef struct {
          uint32_t   nodeId;
          uint32_t   swPhysicalId;
          uint32_t   swPort;
          uint32_t   rlanId;
          bool       connectToGpu;
          uint32_t   peerGpuPhysicalId;
          uint32_t   peerGpuPort;
          bool       connectToSw;
          uint32_t   peerSwNodeId;
          uint32_t   peerSwPhysicalId;
          uint32_t   peerSwPort;
      } SwitchPort_t;

      E4700EndpointType_t mSxm4Slot0;   // On E4700, LR port  0 - 11
      E4700EndpointType_t mSxm4Slot1;   // On E4700, LR port 16 - 25
      E4700EndpointType_t mExaMAXslot0; // On E4700, LR port 12 - 15, 32, 33
      E4700EndpointType_t mExaMAXslot1; // On E4700, LR port 26, 27, 30, 31, 34, 35

      bool mEnableWarmup;   // true if access ports can loop packet from the connected GPU back to the same GPU
      bool mEnableSpray;    // true if spray is enabled
      bool mUseTrunkPorts;  // true if using Trunk ports first even GPUs are connected directly
      uint32_t mMaxTargetId;

      int getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPortNum );
      int getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex );
      int getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId);

      void getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                uint32_t *egressPorts, uint32_t *numEgressPorts);

      SwitchPort_t mSwitchPorts[MAX_PORTS_PER_LWSWITCH];
};

#endif
