
#ifndef BASIC_FMODEL_CONFIG1_H
#define BASIC_FMODEL_CONFIG1_H

#include "fabricConfig.h"

#define BASIC_FMODEL1_NUM_NODES     1
#define BASIC_FMODEL1_NUM_LIMEROCKS 1
#define BASIC_FMODEL1_NUM_GPUS      2



class basicFModelConfig1 : public fabricConfig
{
public:
      basicFModelConfig1( fabricTopologyEnum topo );
      virtual ~basicFModelConfig1();

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
