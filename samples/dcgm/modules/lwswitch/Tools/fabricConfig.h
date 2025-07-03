#ifndef FABRIC_CONFIG_H
#define FABRIC_CONFIG_H

#include "DcgmFMCommon.h"
#include "topology.pb.h"
#include "logging.h"
#include <g_lwconfig.h>

#define NUM_PORTS_PER_WILLOW           18
#define NUM_LWLINKS_PER_GPU             6
#define NUM_INGR_REQ_ENTRIES_PER_GPU    4    // up to 4 entries, each entry represents a 16G memory region
#define NUM_INGR_RESP_ENTRIES_PER_GPU   NUM_LWLINKS_PER_GPU

#define INGRESS_REQ_TABLE_SIZE          8192
#define INGRESS_RESP_TABLE_SIZE         8192

#define MAX_NUM_NODES                   8
#define MAX_NUM_GPUS_PER_NODE           16
#define MAX_NUM_WILLOWS_PER_NODE        12

#define NUM_PORTS_PER_LR                36

#define INGRESS_REMAP_TABLE_SIZE        2048
#define INGRESS_RID_TABLE_SIZE          512
#define INGRESS_RLAN_TABLE_SIZE         512
#define INGRESS_RID_MAX_PORTS           16
#define INGRESS_RLAN_MAX_GROUPS         16

#define LR_MAX_NUM_GPUS_PER_NODE        16
#define LR_MAX_NUM_LRS_PER_NODE         48

#define FAB_ADDR_RANGE_16G              (1LL << 34)

#define FABRIC_TOPOLOGY_ENUM(TOPOLOGY) \
    TOPOLOGY(UNKNOWN_TOPOLOGY)    \
    TOPOLOGY(BASIC_E3600_CONFIG1) \
    TOPOLOGY(BASIC_E3600_CONFIG2) \
    TOPOLOGY(BASIC_E3600_CONFIG3) \
    TOPOLOGY(BASIC_E3600_CONFIG4) \
    TOPOLOGY(BASIC_E3600_CONFIG5) \
    TOPOLOGY(BASIC_E3600_CONFIG6) \
    TOPOLOGY(BASIC_E3600_CONFIG7) \
    TOPOLOGY(BASIC_E3600_CONFIG8) \
    TOPOLOGY(BASIC_E3600_CONFIG9) \
    TOPOLOGY(VC_FLIP)             \
    TOPOLOGY(VANGUARD_CONFIG)     \
    TOPOLOGY(EXPLORER8_CONFIG)    \
    TOPOLOGY(DGX2_CONFIG)         \
    TOPOLOGY(EXPLORER2_CONFIG)    \
    TOPOLOGY(EXPLORER8LB_CONFIG)  \
    TOPOLOGY(EXPLORER8BW_CONFIG)  \
    TOPOLOGY(EXPLORER8SL_CONFIG)  \
    TOPOLOGY(EXPLORER_LOOP)       \
    TOPOLOGY(HGX2_BASEBOARD1_LOOP)     \
    TOPOLOGY(HGX2_BASEBOARD2_LOOP)     \
    TOPOLOGY(HGX2_TWO_BASEBOARDS_LOOP) \
    TOPOLOGY(EMULATION_CONFIG)    \
    TOPOLOGY(DGX2_KT_2VM_CONFIG)       \
    TOPOLOGY(LR_EMULATION_CONFIG) \
    TOPOLOGY(DGX2_TRUNK_SPRAY_CONFIG)  \
    TOPOLOGY(MAX_TOPOLOGY_CONFIG)

#define MAKE_ENUM(VAR) VAR,
typedef enum {
    FABRIC_TOPOLOGY_ENUM(MAKE_ENUM)
} fabricTopologyEnum;

class fabricConfig
{
public:
       fabricConfig( fabricTopologyEnum topology );
       virtual ~fabricConfig();

void makeOneGpu( int nodeIndex,     int gpuIndex,  int endpointID,  int peerID,
                 int peerIdPortMap, int logicalToPhyPortMap,
                 int64_t fabricAddr,int64_t fabricAddrRange, int gpuPhysicalID);

void makeOneGpu( int nodeIndex,     int gpuIndex,  int endpointID,  int peerID,
                 int peerIdPortMap, int logicalToPhyPortMap,
                 int64_t gpaFabricAddr,int64_t gpaFabricAddrRange,
                 int64_t flaFabricAddr,int64_t flaFabricAddrRange,
                 int gpuPhysicalID);

void makeOneIngressReqEntry( int nodeIndex, int willowIndex, int portIndex,
                             int index, int64_t address, int routePolicy,
                             int vcModeValid7_0, int vcModeValid15_8,
                             int vcModeValid17_16, int entryValid);

void makeOneIngressRespEntry( int nodeIndex, int willowIndex, int portIndex,
                              int index, int routePolicy, int vcModeValid7_0,
                              int vcModeValid15_8, int vcModeValid17_16,
                              int entryValid);

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void makeOneRemapEntry( int nodeIndex, int swIndex, int portIndex,
                        int index, int entryValid, int64_t address,
                        int reqContextChk, int reqContextMask, int reqContextRep,
                        int addressOffset, int addressBase, int addressLimit,
                        int targetId, int routingFunction, int irlSelect,
                        int p2rSwizEnable, int mult2, int planeSelect);

void makeOneRIDRouteEntry( int nodeIndex, int swIndex, int portIndex,
                           int index, int entryValid, int rMod,
                           int portCount, int *vcMap, int *egressPort);


void makeOneRLANRouteEntry( int nodeIndex, int swIndex, int portIndex,
                            int index, int entryValid, int groupCount,
                            int *groupSelect, int *groupSize);
#endif

void makeOneGangedLinkEntry( int nodeIndex, int willowIndex, int portIndex,
                             int index, int data);

void makeOneAccessPort( int nodeIndex, int willowIndex, int portIndex,
                        int farNodeID, int farPeerID,   int farPortNum,
                        PhyMode phyMode);

void makeOneAccessPort( int nodeIndex, int willowIndex, int portIndex,
                        int farNodeID, int farPeerID,   int farPortNum,
                        PhyMode phyMode, uint32_t requesterLinkID);

void makeOneTrunkPort( int nodeIndex, int willowIndex, int portIndex,
                       int farNodeID, int farSwitchID, int farPortNum,
                       PhyMode phyMode );

virtual void makeOneNode( int nodeIndex, int gpuNum, int willowNum ) = 0;
virtual void makeNodes() = 0;
virtual void makeOneWillow( int nodeIndex, int willowIndex ) = 0;
virtual void makeIngressReqTable( int nodeIndex, int willowIndex ) = 0;
virtual void makeIngressRespTable( int nodeIndex, int willowIndex ) = 0;
virtual void makeGangedLinkTable( int nodeIndex, int willowIndex ) = 0;
virtual void makeAccessPorts( int nodeIndex, int willowIndex ) = 0;
virtual void makeTrunkPorts( int nodeIndex, int willowIndex ) = 0;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
virtual void makeOneLwswitch( int nodeIndex, int swIndex ) = 0;
virtual void makeRemapTable( int nodeIndex, int swIndex ) = 0;
virtual void makeRIDRouteTable( int nodeIndex, int swIndex ) = 0;
virtual void makeRLANRouteTable( int nodeIndex, int swIndex ) = 0;
#endif

public:
    fabric               *mFabric;
    fabricTopologyEnum    fabricTopology;

    const char *getFabricTopologyName(fabricTopologyEnum topology);

protected:
    node                 *nodes[MAX_NUM_NODES];
    lwSwitch             *switches[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE];
    GPU                  *gpus[MAX_NUM_NODES][MAX_NUM_GPUS_PER_NODE];
    peerIDPortMap        *gpuPeerIdMaps[MAX_NUM_NODES][MAX_NUM_GPUS_PER_NODE][NUM_LWLINKS_PER_GPU];
    accessPort           *accesses[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW];
    trunkPort            *trunks[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW];
    ingressRequestTable  *reqEntry[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW][INGRESS_REQ_TABLE_SIZE];
    ingressResponseTable *respEntry[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW][INGRESS_RESP_TABLE_SIZE];
    int32_t              *gangedLinkEntry[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW][GANGED_LINK_TABLE_SIZE];
    switchPortConfig     *portConfigs[MAX_NUM_NODES][MAX_NUM_WILLOWS_PER_NODE][NUM_PORTS_PER_WILLOW];

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
    rmapPolicyEntry      *rmapEntry[MAX_NUM_NODES][LR_MAX_NUM_LRS_PER_NODE][NUM_PORTS_PER_LR][INGRESS_REMAP_TABLE_SIZE];
    ridRouteEntry        *ridEntry[MAX_NUM_NODES][LR_MAX_NUM_LRS_PER_NODE][NUM_PORTS_PER_LR][INGRESS_RID_TABLE_SIZE];
    rlanRouteEntry       *rlanEntry[MAX_NUM_NODES][LR_MAX_NUM_LRS_PER_NODE][NUM_PORTS_PER_LR][INGRESS_RLAN_TABLE_SIZE];
#endif

    uint64_t gpuFabricAddrBase[MAX_NUM_NODES * MAX_NUM_GPUS_PER_NODE];
    uint64_t gpuFabricAddrRange[MAX_NUM_NODES * MAX_NUM_GPUS_PER_NODE];
    uint64_t gpuFlaAddrBase[MAX_NUM_NODES * MAX_NUM_GPUS_PER_NODE];
    uint64_t gpuFlaAddrRange[MAX_NUM_NODES * MAX_NUM_GPUS_PER_NODE];

    void setFabricTopologyName();
    void setFabricTopologyTime();

private:

};

#endif
