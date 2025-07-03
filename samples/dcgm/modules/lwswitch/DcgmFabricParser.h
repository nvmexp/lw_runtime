#ifndef DCGM_FABRIC_PARSER_H
#define DCGM_FABRIC_PARSER_H


#include "g_lwconfig.h"
#include "DcgmFMError.h"
#include "DcgmFMCommon.h"
#include "topology.pb.h"
#include "fabricmanager.pb.h"
#include "LwcmSettings.h"
#include "lwml_internal.h"

typedef struct NodeKeyType
{
    uint32_t nodeId;

    bool operator<(const NodeKeyType &k) const {
        return (nodeId < k.nodeId);
    }
} NodeKeyType;

typedef struct NodeConfig
{
    uint32_t     nodeId;
    std::string *IPAddress;
    nodeSystemPartitionInfo partitionInfo;
} NodeConfig;

typedef struct SwitchKeyType
{
    uint32_t nodeId;
    uint32_t physicalId;

    bool operator<(const SwitchKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) );
    }
} SwitchKeyType;

typedef struct GpuKeyType
{
    uint32_t nodeId;
    uint32_t physicalId;

    bool operator<(const GpuKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) );
    }
} GpuKeyType;

typedef struct PortKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t physicalId;

    bool operator<(const PortKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) );
    }
} PortKeyType;

typedef struct ReqTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const ReqTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index < k.index) ) );
    }
} ReqTableKeyType;

typedef struct RespTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RespTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RespTableKeyType;

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

typedef struct RmapTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RmapTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index < k.index) ) );
    }
} RmapTableKeyType;

typedef struct RidTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RidTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RidTableKeyType;

typedef struct RlanTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const RlanTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} RlanTableKeyType;

#endif

typedef struct GangedLinkTableKeyType
{
    uint32_t  nodeId;
    uint32_t  portIndex;
    uint32_t  physicalId;
    uint32_t  index;

    bool operator<(const GangedLinkTableKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex ==  k.portIndex) && (index <  k.index) ) );
    }
} GangedLinkTableKeyType;

typedef struct TopologyLWLinkConnEndPoint
{
    uint32_t nodeId;
    uint32_t lwswitchOrGpuId;
    uint32_t portIndex;

    bool operator==(const TopologyLWLinkConnEndPoint& rhs)
    {
        if ( (nodeId == rhs.nodeId) &&
             (lwswitchOrGpuId == rhs.lwswitchOrGpuId) &&
             (portIndex == rhs.portIndex) ) {
            return true;
        } else {
            return false;
        }
    }
} TopologyLWLinkConnEndPoint;

typedef struct TopologyLWLinkConn
{
    TopologyLWLinkConnEndPoint localEnd;
    TopologyLWLinkConnEndPoint farEnd;
    uint8_t  connType; //ACCESS_XX vs TRUNK
} TopologyLWLinkConn;

// list of connections for a node
typedef std::list<TopologyLWLinkConn> TopologyLWLinkConnList;
// map of connections for all the nodes
typedef std::map<uint32, TopologyLWLinkConnList> TopologyLWLinkConnMap;

class DcgmFabricParser
{
public:
    typedef enum PruneFabricType {
        PRUNE_ALL = 0,
        PRUNE_SWITCH,
        PRUNE_GPU
    } PruneFabricType;

    DcgmFabricParser();
    virtual ~DcgmFabricParser();

    FM_ERROR_CODE parseFabricTopology( const char *topoFile );
    FM_ERROR_CODE parseFabricTopologyConf( const char *topoConfFile );
    void fabricParserCleanup();
    bool isSwtichGpioPresent();

    void disableGpu( GpuKeyType &gpu );

    void disableSwitch( SwitchKeyType &key );

    void modifyFabric(PruneFabricType pruneType);
    int getNumDisabledGpus();
    int getNumDisabledSwitches();

    uint32_t getNodeId(uint32_t nodeIndex);
    uint32_t getNodeIndex(uint32_t nodeId);

    accessPort *getAccessPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );
    trunkPort *getTrunkPortInfo( uint32_t nodeId, uint32_t physicalId, uint32_t portNum );

    const char *getPlatformId();

    std::map <NodeKeyType,      NodeConfig *>               NodeCfg;
    std::map <SwitchKeyType,    lwswitch::switchInfo *>     lwswitchCfg;
    std::map <PortKeyType,      lwswitch::switchPortInfo *> portInfo;
    std::map <ReqTableKeyType,  ingressRequestTable  *>     reqEntry;
    std::map <RespTableKeyType, ingressResponseTable *>     respEntry;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 
    std::map <RmapTableKeyType,   rmapPolicyEntry *>            rmapEntry;
    std::map <RidTableKeyType,    ridRouteEntry *>              ridEntry;
    std::map <RlanTableKeyType,   rlanRouteEntry *>             rlanEntry;
#endif
    std::map <GangedLinkTableKeyType, int32_t *>            gangedLinkEntry;
    std::map <GpuKeyType,       lwswitch::gpuInfo * >       gpuCfg;

    TopologyLWLinkConnMap    lwLinkConnMap;

private:

    fabric               *mpFabric;
    lwmlChipArchitecture_t mArch;

    std::set <PortKeyType> mLoopbackPorts;
    std::set <uint32_t> mDisabledGpuEndpointIds;
    std::set <SwitchKeyType> mDisabledSwitches;
    std::set <uint32_t> mIlwalidIngressReqEntries;
    std::set <uint32_t> mIlwalidIngressRespEntries;
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 
    std::set <uint32_t> mIlwalidIngressRmapEntries;
    std::set <uint32_t> mIlwalidIngressRidEntries;
    std::set <uint32_t> mIlwalidIngressRlanEntries;
#endif

    FM_ERROR_CODE parseOneNode( const node &node, int nodeIndex );
    FM_ERROR_CODE parseOneLwswitch( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseAccessPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseTrunkPorts( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseOnePort(const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId,
                      int portIndex, int isAccess);
    FM_ERROR_CODE parseOneGpu( const GPU &gpu, uint32_t nodeId, int gpuIndex );

    FM_ERROR_CODE parseOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const ingressRequestTable &cfg);
    FM_ERROR_CODE parseOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const ingressResponseTable &cfg);
    FM_ERROR_CODE parseOneGangedLinkEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const gangedLinkTable &cfg);

    FM_ERROR_CODE parseIngressReqTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseIngressRespTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

    FM_ERROR_CODE parseOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                           const rmapPolicyEntry &cfg);
    FM_ERROR_CODE parseOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const ridRouteEntry &cfg);
    FM_ERROR_CODE parseOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                            const rlanRouteEntry &cfg);

    FM_ERROR_CODE parseIngressRmapTable( const lwSwitch &lwSwitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseIngressRidTable( const lwSwitch &lwSwitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseIngressRlanTable( const lwSwitch &lwSwitch, uint32_t nodeId, uint32_t physicalId );

#endif

    FM_ERROR_CODE parseGangedLinkTable( const lwSwitch &lwswitch, uint32_t nodeId, uint32_t physicalId );
    FM_ERROR_CODE parseLWLinkConnections( const node &node, uint32_t nodeId );

    FM_ERROR_CODE parseDisableGPUConf( std::vector<std::string> &gpus );
    FM_ERROR_CODE parseDisableSwitchConf( std::vector<std::string> &switches );
    FM_ERROR_CODE parseLoopbackPorts( std::vector<std::string> &ports );

    void modifyOneIngressReqEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                   ingressRequestTable *entry );
    void modifyOneIngressRespEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    ingressResponseTable *entry );
    void modifyRoutingTable();

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
// new routing tables introduced with LimeRock 

    void modifyOneIngressRmapEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                   rmapPolicyEntry *entry );
    void modifyOneIngressRidEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    ridRouteEntry *entry );
    void modifyOneIngressRlanEntry( uint32_t nodeId, uint32_t physicalId, uint32_t localPortNum,
                                    rlanRouteEntry *entry );

#endif

    void pruneOneAccessPort( uint32_t nodeId, uint32_t physicalId,
                             lwSwitch *pLwswitch, uint32_t localportNum );
    void pruneAccessPorts();

    void pruneOneGpu( uint32_t gpuEndpointID );
    void pruneGpus();

    void pruneOneTrunkPort( uint32_t nodeId, uint32_t physicalId,
                            lwSwitch *pLwswitch, uint32_t localportNum );
    void pruneTrunkPorts();

    void pruneOneSwitch( SwitchKeyType &key );
    void pruneSwitches();

    int checkLWLinkConnExists( TopologyLWLinkConnList &connList,
                               TopologyLWLinkConn newConn );

    void copyTrunkPortConnInfo( trunkPort trunkPort, uint32_t nodeId,
                                uint32_t physicalId,int portIndex, 
                                TopologyLWLinkConnList &connList );

    void copyAccessPortConnInfo( accessPort accessPort, uint32_t nodeId,
                                 uint32_t physicalId,int portIndex, 
                                 TopologyLWLinkConnList &connList );
};

#endif
