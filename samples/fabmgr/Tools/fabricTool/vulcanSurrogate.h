
#ifndef VULCAN_SURROGATE_H
#define VULCAN_SURROGATE_H

#include "fabricConfig.h"

class vulcanSurrogate : public fabricConfig
{
public:
      vulcanSurrogate( fabricTopologyEnum topo );

      virtual ~vulcanSurrogate();

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

      void setConfig( bool enableTrunkLookback, bool enableSpray, const char *sharedPartitionJsonFile );

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

      bool mEnableTrunkLookback;   // true if use trunk loop back, for MODS and factory diag tests with densilink connectors
      bool mEnableSpray;          // true if spray is enabled
      uint32_t mMaxTargetId;
      uint32_t mNumSwitches;
      uint32_t mNumGpus;

      int getSwitchPortIndex( uint32_t nodeIndex, uint32_t swIndex, uint32_t swPortNum );
      int getNumTrunkPorts( uint32_t nodeIndex, uint32_t swIndex );
      int getSwitchPortIndexWithRlanId( uint32_t nodeIndex, uint32_t swIndex, uint32_t rlanId);
      bool isGpuConnectedToSwitch( uint32_t swNodeId, uint32_t swIndex, uint32_t gpuNodeId, uint32_t gpuPhysicalId );

      void getEgressPortsToGpu( uint32_t nodeIndex, uint32_t swIndex, uint32_t ingressPortNum,
                                uint32_t destGpuNodeId, uint32_t destGpuPhysicalId,
                                uint32_t *egressPorts, uint32_t *numEgressPorts, bool *isLastHop );

      void fillSystemPartitionInfo(int nodeIndex);
      void fillSharedPartInfoTable(int nodeIndex);
      void updateSharedPartInfoLinkMasks( SharedPartInfoTable_t &partInfo );
      bool getSwitchIndexByPhysicalId( uint32_t physicalId, uint32_t &swIndex );

      SwitchPort_t mSwitchPorts[MAX_NUM_LWSWITCH_PER_NODE][MAX_PORTS_PER_LWSWITCH];

      #define VULCAN_SURROGATE_NUM_SHARED_LWSWITCH_FABRIC_PARTITIONS  0
      #define VULCAN_SURROGATE_NUM_TRUNK_LINKS 96
};

#endif
