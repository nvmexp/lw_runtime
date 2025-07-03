#ifndef FABRIC_TOOL_H
#define FABRIC_TOOL_H

#define OPT_DISABLE_GPU                "--disable-gpu"
#define OPT_DISABLE_SWITCH             "--disable-switch"
#define OPT_DISABLE_PORT               "--disable-port"
#define OPT_PORT_LOOPBACK              "--loopback-port"


typedef struct GpuKeyType
{
    uint32_t nodeId;
    uint32_t gpuPhysicalId;

    bool operator<(const GpuKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (gpuPhysicalId <  k.gpuPhysicalId) ) );
    }
} GpuKeyType;

typedef struct SwitchKeyType
{
    uint32_t nodeId;
    uint32_t physicalId;

    bool operator<(const SwitchKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) );
    }
} SwitchKeyType;

struct SwitchComp
{
bool operator() (const SwitchKeyType& lhs, const SwitchKeyType& rhs) const {
        return ( ( lhs.nodeId < rhs.nodeId) ||
                 ( (lhs.nodeId ==  rhs.nodeId) && (lhs.physicalId <  rhs.physicalId) ) );
    }
};

typedef struct PortKeyType
{
    uint32_t nodeId;
    uint32_t portIndex;
    uint32_t physicalId;

    bool operator<(const PortKeyType &k) const {
        return ( (nodeId < k.nodeId) ||
                 ( (nodeId ==  k.nodeId) && (physicalId <  k.physicalId) ) ||
                 ( (nodeId ==  k.nodeId) && (physicalId ==  k.physicalId) &&
                   (portIndex <  k.portIndex) ) );
    }
} PortKeyType;

struct PortComp
{
bool operator() (const PortKeyType& lhs, const PortKeyType& rhs) const {
        return ( ( lhs.nodeId < rhs.nodeId) ||
                 ( (lhs.nodeId ==  rhs.nodeId) && (lhs.physicalId <  rhs.physicalId) ) ||
                 ( (lhs.nodeId ==  rhs.nodeId) && (lhs.physicalId ==  rhs.physicalId) &&
                 (lhs.portIndex <  rhs.portIndex) ) );
    }
};

#endif
