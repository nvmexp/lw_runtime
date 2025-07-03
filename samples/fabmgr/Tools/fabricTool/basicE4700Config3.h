
#ifndef BASIC_E4700_CONFIG3_H
#define BASIC_E4700_CONFIG3_H

#include "fabricConfig.h"

class basicE4700Config3 : public fabricConfig
{
public:
    typedef enum {
        NotPopulated  = 0,
        PG506,       // GA100 in SXM4 slot
        E4702,       // Loop back in SXM4 slot
        E4705        // Loop back in ExaMAX slot
    } E4700EndpointType_t;

      basicE4700Config3( fabricTopologyEnum topo );

      virtual ~basicE4700Config3();

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

      void setE4700Config( bool enableWarmup, bool enableSpray );

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

      bool mEnableWarmup;   // true if access ports can loop packet from the connected GPU back to the same GPU
      bool mEnableSpray;    // true if spray is enabled
      uint32_t mMaxTargetId;
      uint32_t mNumSwitches;
      uint32_t mNumGpus;

      SwitchPort_t mSwitchPorts[2][MAX_PORTS_PER_LWSWITCH];
};

#endif
