
#ifndef LS_FSF_CONFIG_H
#define LS_FSF_CONFIG_H

#include "fabricConfig.h"

class lsFsfConfig : public fabricConfig
{
public:
      lsFsfConfig( fabricTopologyEnum topo );
      virtual ~lsFsfConfig();

      virtual void    makeOneNode( int nodeIndex, int gpuNum, int lsNum );
                                              // make one node, and add the GPUs, NPUs and LSs
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
      uint32_t mNumGpus;
      uint32_t mNumSwitches;
};

#endif
