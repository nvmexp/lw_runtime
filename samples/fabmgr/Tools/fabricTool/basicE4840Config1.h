
#ifndef BASIC_E4840_CONFIG1_H
#define BASIC_E4840_CONFIG1_H

#include "fabricConfig.h"

class basicE4840Config1 : public fabricConfig
{
public:
    typedef enum {
        NotPopulated  = 0,
        PG520,       // GA100 in SXM5 slot
        E4824        // Loop out in SXM5
    } E4840EndpointType_t;

      basicE4840Config1( fabricTopologyEnum topo );

      virtual ~basicE4840Config1();

      virtual void makeOneNode( int nodeIndex, int gpuNum, int LSNum );
                                           // make one node, and add the GPUs, NPUs and LSs
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

      void setE4840Config(E4840EndpointType_t sxm5Slot0, E4840EndpointType_t sxm5Slot1,
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

      E4840EndpointType_t mSxm5Slot0;   // On E4840, LS port  0 - 17
      E4840EndpointType_t mSxm5Slot1;   // On E4840, LS port 18 - 35

      bool mEnableWarmup;   // true if access ports can loop packet from the connected GPU back to the same GPU
      bool mEnableSpray;    // true if spray is enabled
      bool mUseTrunkPorts;  // true if using Trunk ports first even GPUs are connected directly
      uint32_t mNumSwitches;
      uint32_t mNumGpus;

      int getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPortNum );
      int getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex );
      int getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId);

      void getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                uint32_t *egressPorts, uint32_t *numEgressPorts, bool *isLastHop );

      SwitchPort_t mSwitchPorts[MAX_NUM_LWSWITCH_PER_NODE][MAX_PORTS_PER_LWSWITCH];
};

#endif
