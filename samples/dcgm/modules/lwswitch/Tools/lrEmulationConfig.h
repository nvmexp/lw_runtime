
#ifndef LR_EMULATION_CONFIG_H
#define LR_EMULATION_CONFIG_H

#include "fabricConfig.h"

class lrEmulationConfig : public fabricConfig
{
public:
      lrEmulationConfig( fabricTopologyEnum topo );
      virtual ~lrEmulationConfig();

      virtual void    makeOneNode( int nodeIndex, int gpuNum, int lrNum );
                                              // make one node, and add the GPUs, NPUs and LRs
      virtual void    makeNodes();            // make the nodes

      virtual void    makeOneLwswitch( int nodeIndex, int swIndex );

      virtual void makeRemapTable( int nodeIndex, int swIndex );

      virtual void makeRIDRouteTable( int nodeIndex, int swIndex );

      virtual void makeRLANRouteTable( int nodeIndex, int swIndex );

      virtual void makeGangedLinkTable( int nodeIndex, int swIndex );

      virtual void makeAccessPorts( int nodeIndex, int swIndex );

      virtual void makeTrunkPorts( int nodeIndex, int swIndex );

      virtual void  makeOneWillow( int nodeIndex, int willowIndex );

      virtual void makeIngressReqTable( int nodeIndex, int willowIndex );

      virtual void makeIngressRespTable( int nodeIndex, int willowIndex );

private:

};

#endif
